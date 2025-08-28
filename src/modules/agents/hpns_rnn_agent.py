import torch as th          # 导入 PyTorch 库，th 是别名，用于张量计算
import torch.nn as nn       # 导入神经网络模块，用于定义层结构
import torch.nn.functional as F  # 导入激活函数和操作函数
import math                 # 导入数学函数，用于权重初始化
from torch.nn.parameter import Parameter  # 导入可训练参数类

# 定义 Kaiming 均匀初始化函数，优化权重和偏置的初始值
def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)  # 计算输入/输出神经元数量（fan_in 或 fan_out）
    std = gain / math.sqrt(fan)  # 根据 Kaiming 初始化计算标准差
    bound_w = math.sqrt(3.0) * std  # 计算权重均匀分布的上限和下限
    bound_b = 1 / math.sqrt(fan)  # 计算偏置均匀分布的上限和下限
    with th.no_grad():  # 不计算梯度，防止初始化影响训练
        tensor_w.data.uniform_(-bound_w, bound_w)  # 权重初始化为均匀分布
        if tensor_b is not None:  # 如果有偏置
            tensor_b.data.uniform_(-bound_b, bound_b)  # 偏置初始化为均匀分布
    # 作用：为神经网络层提供合理的初始权重，加速收敛
    # 论文联系：页面7，HPN 使用 hypernet，初始化影响表示能力
    # 类比：像给厨具上油，确保启动时顺畅


# 定义 Merger 类，用于融合多头输出
class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()  # 调用父类 nn.Module 初始化，注册参数
        self.head = head  # 保存多头数量
        if head > 1:  # 如果多头大于1，启用加权融合
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))  # 创建可训练权重张量，初始值全1
            self.softmax = nn.Softmax(dim=1)  # 沿多头维度归一化权重

    def forward(self, x):
        """
        :param x: 输入张量，形状 [batch_size, n_head, fea_dim]，多头输出特征
        :return: 输出张量，形状 [batch_size, fea_dim]，融合后的单一特征
        """
        if self.head > 1:  # 多头情况，执行加权融合
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)  # 按权重加和，沿多头维度
        else:  # 单头情况，直接挤压维度
            return th.squeeze(x, dim=1)  # 移除多头维度
        # 作用：融合多头增强表示，兼容单头优化
        # 论文联系：页面7，unify_heads 融合多头输出
        # 类比：像“投票系统”，多头投票决定权重，求和得出共识


# 定义 HPNS_RNNAgent 类，HPN 变体的 RNN 代理
class HPNS_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        """
        构造函数：初始化 HPNS_RNNAgent 的网络结构和参数
        :param input_shape: 输入形状，分解为 [自身维度, (敌方实体数, 特征维度), (友方实体数, 特征维度)]
        :param args: 配置参数，包含 n_agents, n_enemies, n_heads 等
        初始化包括嵌入表、线性层、hypernetworks 和 RNN
        论文页面7：定义 HPN 的输入/输出 hypernetworks 和 RNN 骨干
        类比：像组装机器人，添加零件 (层) 和配置 (args)
        """
        super(HPNS_RNNAgent, self).__init__()  # 调用父类 nn.Module 初始化，注册参数
        self.args = args  # 保存配置参数，便于类内访问
        self.n_agents = args.n_agents  # 智能体数量
        self.n_allies = args.n_allies  # 友方实体数量
        self.n_enemies = args.n_enemies  # 敌方实体数量
        self.n_actions = args.n_actions  # 动作数量
        self.n_heads = args.hpn_head_num  # 多头数量
        self.rnn_hidden_dim = args.rnn_hidden_dim  # RNN 隐藏层维度

        # 分解输入形状：[自身维度, (敌方实体数, 特征维度), (友方实体数, 特征维度)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # 提取敌方特征维度 (忽略实体数)
        self.ally_feats_dim = self.ally_feats_dim[-1]  # 提取友方特征维度

        if self.args.obs_agent_id:  # 如果观察包含智能体 ID
            # 创建嵌入表，将智能体 ID 映射为向量
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:  # 如果观察包含上一动作
            # 创建嵌入表，将动作 ID 映射为向量
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # 自身特征线性层 (无需 hypernet，唯一特征)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # 线性层：自身特征到隐藏维度

        # 输入层 hypernetworks (PI 部分)，处理多实体确保置换不变性
        self.hyper_enemy = nn.Sequential(  # 敌方特征处理网络
            nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),  # 第一层：输入到隐藏层
            nn.ReLU(inplace=True),  # 激活函数，增加非线性
            nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)  # 输出多头权重
        )  # 输出形状：(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * n_heads

        if self.args.map_type == "MMM":  # 如果是 MMM 地图，特殊处理
            assert self.n_enemies >= self.n_agents, "For MMM map, n_enemies must >= n_agents due to SMAC ID conflict"
            self.hyper_ally = nn.Sequential(  # 友方特征处理网络
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, ((self.ally_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # 输出形状：包含偏置和多头权重，用于救援动作
            self.unify_output_heads_rescue = Merger(self.n_heads, 1)  # 融合救援动作多头输出

        else:  # 非 MMM 地图
            self.hyper_ally = nn.Sequential(  # 友方特征处理网络
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim * self.n_heads)
            )  # 输出形状：仅权重，无偏置

        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)  # 融合输入层多头嵌入
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)  # RNN 单元，处理序列输入
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # 输出正常动作 Q 值 (PI)
        self.unify_output_heads = Merger(self.n_heads, 1)  # 融合攻击动作多头输出

        # 重置 hypernet 参数 (可选，未启用)
        # self._reset_hypernet_parameters(init_type="xavier")
        # self._reset_hypernet_parameters(init_type="kaiming")

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        """
        重置 hypernet 层的参数，优化初始值
        :param init_type: 初始化类型，'kaiming' 或 'xavier'
        使用 Kaiming 或 Xavier 初始化权重和偏置
        论文页面7：初始化影响 hypernet 表示能力
        类比：像为厨具上油，确保启动顺畅
        """
        gain = 2 ** (-0.5)  # Kaiming 初始化增益
        # 输入层 hypernet 参数重置
        for m in self.hyper_enemy.modules():  # 遍历 hyper_enemy 的子模块
            if isinstance(m, nn.Linear):  # 如果是线性层
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)  # Kaiming 初始化
                else:
                    nn.init.xavier_normal_(m.weight.data)  # Xavier 初始化
                    m.bias.data.fill_(0.)  # 偏置置零
        for m in self.hyper_ally.modules():  # 遍历 hyper_ally 的子模块
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)

    def init_hidden(self):
        """
        初始化 RNN 隐藏状态
        返回：全零张量，形状 [1, rnn_hidden_dim]，与模型设备一致
        论文页面5：RNN 处理部分可观测输入需要初始状态
        类比：像清空记忆本，episode 开始时重置
        """
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()  # 创建全零张量

    def forward(self, inputs, hidden_state):
        """
        前向传播函数：从输入观察计算 Q 值和新隐藏状态
        :param inputs: 输入元组 (bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices)
        :param hidden_state: 上一步隐藏状态
        :return: Q 值 [bs, n_agents, n_actions], 新隐藏状态 [bs, n_agents, rnn_hidden_dim]
        论文页面7：对应 HPN 数据流：输入 hypernet → 嵌入 → RNN → 输出 hypernet
        类比：像生产流水线，输入原料 (观察) → 处理 (hypernet + pooling) → 记忆 (RNN) → 输出 (Q 值)
        """
        # 解包输入，bs 是 batch size，own_feats_t 是自身特征
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) 自身特征嵌入
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]，线性变换自身特征

        # (2) ID 嵌入
        if self.args.obs_agent_id:  # 如果包含智能体 ID
            agent_indices = embedding_indices[0]  # 提取 ID 索引
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)  # 添加 ID 嵌入
        if self.args.obs_last_action:  # 如果包含上一动作
            last_action_indices = embedding_indices[-1]  # 提取动作索引
            if last_action_indices is not None:  # t != 0
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(-1, self.rnn_hidden_dim)  # 添加动作嵌入

        # (3) 敌方特征嵌入 (PI 实现)
        hyper_enemy_out = self.hyper_enemy(enemy_feats_t)  # [bs * n_agents * n_enemies, ...]，生成敌方权重和偏置
        fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(  # 提取权重部分
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(  # 矩阵乘法计算嵌入
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]，sum pooling 实现 PI

        # (4) 友方特征嵌入 (PI 实现)
        hyper_ally_out = self.hyper_ally(ally_feats_t)  # 生成友方权重和偏置
        if self.args.map_type == "MMM":
            fc1_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(  # 提取权重部分
                -1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads
            )
        else:
            fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(  # 矩阵乘法计算嵌入
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, n_heads, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]，sum pooling 实现 PI

        # 最终嵌入
        embedding = embedding_own + self.unify_input_heads(  # 融合自身 + 敌方 + 友方嵌入
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, rnn_hidden_dim]

        x = F.relu(embedding, inplace=True)  # 应用 ReLU 激活，增加非线性
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)  # 重塑隐藏状态
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]，RNN 更新隐藏状态

        # 正常动作 Q 值 (PI 输出)
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]，输出正常动作 Q 值

        # 攻击动作 Q 值 (PE 输出)
        fc2_w_attack = hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(  # 提取攻击权重
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
        ).transpose(1, 2).reshape(  # 重塑为 [bs * n_agents, rnn_hidden_dim, n_enemies * n_heads]
            bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
        )
        fc2_b_attack = hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_enemies * self.n_heads)  # 提取偏置
        q_attacks = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(  # 计算多头 Q 值
            bs * self.n_agents * self.n_enemies, self.n_heads, 1
        )
        q_attack = self.unify_output_heads(q_attacks).view(  # 融合多头，输出攻击 Q 值
            bs, self.n_agents, self.n_enemies
        )

        # MMM 地图的救援动作 (PE 实现)
        if self.args.map_type == "MMM":
            fc2_w_rescue = hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(  # 提取救援权重
                bs * self.n_agents, self.n_allies, self.rnn_hidden_dim, self.n_heads
            ).transpose(1, 2).reshape(  # 重塑为 [bs * n_agents, rnn_hidden_dim, n_allies * n_heads]
                bs * self.n_agents, self.rnn_hidden_dim, self.n_allies * self.n_heads
            )
            fc2_b_rescue = hyper_ally_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_allies * self.n_heads)  # 提取偏置
            q_rescues = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(  # 计算多头救援 Q 值
                bs * self.n_agents * self.n_allies, self.n_heads, 1
            )
            q_rescue = self.unify_output_heads_rescue(q_rescues).view(  # 融合多头，输出救援 Q 值
                bs, self.n_agents, self.n_allies
            )
            right_padding = th.ones_like(q_attack[:, -1:, self.n_allies:], requires_grad=False) * (-9999999)  # 填充无效值
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)  # 修改 Medivac Q 值
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)  # 合并攻击 Q 值

        # 合并两种 Q 值
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]，合并正常和攻击 Q 值
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # 返回 Q 值和新隐藏状态
        # 作用：输出完整 Q 值矩阵和更新后的隐藏状态
        # 论文页面7：输出层生成 Q 值，符合 HPN 的 PE 机制
        # 类比：像生成最终评分表 (Q 值) 和记忆更新 (hh)