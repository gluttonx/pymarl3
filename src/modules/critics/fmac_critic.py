from numpy.core.numeric import True_
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 修正：从 hpn_rnn_agent 导入 Hypernet；从 hpns_rnn_agent 导入 Merger
from modules.agents.hpn_rnn_agent import Hypernet  # HPN 中的超网络
from modules.agents.hpns_rnn_agent import Merger   # HPNS 中的多头融合

class FMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FMACCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # 是否启用 HPN 化 critic（仅改变输入表征，输出仍为标量 V_i）
        self.use_hpn_critic = getattr(self.args, "use_hpn_critic", False)
        self.critic_hidden_dim = args.critic_hidden_dim

        if self.use_hpn_critic:
            # obs_component: [move_dim, (n_enemies, enemy_dim), (n_allies, ally_dim), own_dim]
            move_dim, (self.n_enemies, self.enemy_dim), (self.n_allies, self.ally_dim), own_dim = self.args.obs_component
            self.ctx_dim = move_dim + own_dim
            self.n_heads = getattr(self.args, "hpn_head_num", 1)
            self.hpn_hyper_dim = getattr(self.args, "hpn_hyper_dim", 64)
            self.hpn_activation = getattr(self.args, "hpn_hyper_activation", "relu")

            # 上下文线性
            self.fc_ctx = nn.Linear(self.ctx_dim, self.critic_hidden_dim, bias=True)

            # 敌/友集合的超网络（PI）
            self.hyper_enemy = Hypernet(
                input_dim=self.enemy_dim,
                hidden_dim=self.hpn_hyper_dim,
                main_input_dim=self.enemy_dim,
                main_output_dim=self.critic_hidden_dim,
                activation_func=self.hpn_activation,
                n_heads=self.n_heads,
            )
            self.hyper_ally = Hypernet(
                input_dim=self.ally_dim,
                hidden_dim=self.hpn_hyper_dim,
                main_input_dim=self.ally_dim,
                main_output_dim=self.critic_hidden_dim,
                activation_func=self.hpn_activation,
                n_heads=self.n_heads,
            )
            self.unify_input_heads = Merger(self.n_heads, self.critic_hidden_dim)

        # 输入维度：HPN 模式为嵌入维，否则为扁平 obs (+ id)
        self.input_shape = self._get_input_shape(scheme)

        # 头部 MLP：不拼接动作（与 PPO 的 V(s) 对齐）
        self.fc1 = nn.Linear(self.input_shape, self.critic_hidden_dim)
        self.fc2 = nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim)
        self.v_head = nn.Linear(self.critic_hidden_dim, 1)

        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = None

    # 对齐 PPOLearner 的用法：forward(batch, t) -> [bs, n_agents, 1]
    def forward(self, batch, t=None):
        ts = slice(None) if t is None else t
        x = self._build_inputs(batch, ts)  # [bs, n_agents, in_dim]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        v = self.v_head(x)  # [bs, n_agents, 1]
        return v

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        device = batch.device

        # 取某一时刻观测
        raw_obs_t = batch["obs"][:, t]  # [bs, n_agents, obs_dim]

        if not self.use_hpn_critic:
            inputs = [raw_obs_t]
            if getattr(self.args, "obs_agent_id", False):
                inputs.append(th.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1))
            return th.cat([x.view(bs, self.n_agents, -1) for x in inputs], dim=-1)

        # ===== HPN 化路径：按 HPNMAC 的顺序切分 obs =====
        move_dim = self.ctx_dim - self.args.obs_component[-1]  # ctx_dim = move + own
        own_dim = self.args.obs_component[-1]
        enemy_flat_dim = self.n_enemies * self.enemy_dim
        ally_flat_dim = self.n_allies * self.ally_dim

        move_feats, enemy_feats_flat, ally_feats_flat, own_feats = th.split(
            raw_obs_t, [move_dim, enemy_flat_dim, ally_flat_dim, own_dim], dim=-1
        )
        # 1) 上下文
        ctx = th.cat([move_feats, own_feats], dim=-1).reshape(bs * self.n_agents, -1)
        ctx_emb = self.fc_ctx(ctx)  # [bs*n_agents, H]

        # 2) 敌集合（sum pooling + 多头融合）
        enemy = enemy_feats_flat.reshape(bs * self.n_agents, self.n_enemies, self.enemy_dim)
        W_e = self.hyper_enemy(enemy.reshape(-1, self.enemy_dim))  # [bs*n_agents*nE, eDim, H*heads]
        emb_e = th.matmul(enemy.reshape(-1, 1, self.enemy_dim), W_e)  # [bs*n_agents*nE, 1, H*heads]
        emb_e = emb_e.reshape(bs * self.n_agents, self.n_enemies, self.n_heads, self.critic_hidden_dim).sum(dim=1)

        # 3) 友集合
        ally = ally_feats_flat.reshape(bs * self.n_agents, self.n_allies, self.ally_dim)
        W_a = self.hyper_ally(ally.reshape(-1, self.ally_dim))  # [bs*n_agents*nA, aDim, H*heads]
        emb_a = th.matmul(ally.reshape(-1, 1, self.ally_dim), W_a)  # [bs*n_agents*nA, 1, H*heads]
        emb_a = emb_a.reshape(bs * self.n_agents, self.n_allies, self.n_heads, self.critic_hidden_dim).sum(dim=1)

        # 4) 融合得到每个体嵌入
        emb = ctx_emb + self.unify_input_heads(emb_e + emb_a)  # [bs*n_agents, H]
        return emb.reshape(bs, self.n_agents, self.critic_hidden_dim)

    def _get_input_shape(self, scheme):
        if self.use_hpn_critic:
            return self.critic_hidden_dim
        input_shape = scheme["obs"]["vshape"]
        if getattr(self.args, "obs_agent_id", False):
            input_shape += self.n_agents
        return input_shape