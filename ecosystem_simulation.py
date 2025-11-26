import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import os
import sys

# 设置中文字体 (尝试常见的Windows中文字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SPECIES_NAMES = {}

def get_species_name(s_id):
    if s_id not in SPECIES_NAMES:
        # 生成随机中文名
        chars = "龙虎雀龟鹰狼熊狮鲸鲨莲菊梅兰松竹云风雷电光暗星月日天海山川灵神魔仙圣皇"
        # 随机取2个字
        name = "".join(random.sample(chars, 2))
        # 加上ID后缀以防重名
        SPECIES_NAMES[s_id] = f"{name}-{s_id}"
    return SPECIES_NAMES[s_id]

# 定义生物单元基类
class OrganismUnit:
    __slots__ = ('energy', 'dna', 'supply_target', 'supply_ttl')
    def __init__(self, dna=None):
        self.energy = 10.0 # 初始能量给予一点，避免出生即死
        # DNA: 包含遗传信息
        # light_threshold: 决定复制单元变身时选择光合(>阈值)还是连接(<=阈值)
        # replication_threshold: 决定复制单元繁殖所需的能量阈值
        # fertility_threshold: 决定复制单元变身时是否选择光合单元(需同时满足光照和肥力阈值)
        # mutation_prob: 连接单元变异为复制单元的概率
        # degeneration_prob: 复制单元退化为连接单元的概率
        self.dna = dna if dna is not None else {
            'light_threshold': 0.4, 
            'fertility_threshold': 0.4,
            'replication_threshold': 50.0,
            'mutation_prob': 0.01,
            'degeneration_prob': 0.01,
            'fertility_sacrifice_threshold': 0.1, # 肥力低于此值时，连接单元牺牲自己
            'factory_prob': 0.05, # 变身为工厂的概率
            'factory_fertility_threshold': 0.3, # 肥力低于此值时，可能变身为工厂
            'defense_prob': 0.02, # 变身为防御设施的概率
            'connection_threshold': 3.0, # 判定为枢纽节点的连接数阈值
            'replication_connection_sensitivity': 0.0, # 复制阈值对连接数的敏感度 (-1.0 ~ 1.0)
            'hub_behavior': 0.5, # 枢纽策略: <0.3连接, 0.3-0.7光合, >0.7工厂
            'corridor_behavior': 0.4, # 通道策略: <0.6连接, >0.6光合
            'species_id': 0 # 种群ID
        }
        self.supply_target = None
        self.supply_ttl = 0

    def update(self, node, all_nodes):
        # 基础代谢消耗
        self.energy -= 0.5
        pass

    def respond_to_alert(self, node, enemy_pos):
        """响应警报: 设定补给目标"""
        self.supply_target = enemy_pos
        self.supply_ttl = 15

def get_neighbors(node):
    """获取相连的节点"""
    return node.neighbors

def alert_allies(all_nodes, center_node, species_id, enemy_pos, max_hops=4):
    """警报机制优化: 使用BFS在图上传播警报 (基于跳数而非物理距离)"""
    # BFS 初始化
    queue = [(center_node, 0)]
    visited = {center_node}
    
    while queue:
        curr_node, hops = queue.pop(0)
        
        # 逻辑处理
        unit = curr_node.organism
        if unit and unit.dna.get('species_id') == species_id:
            unit.respond_to_alert(curr_node, enemy_pos)
            
        # 继续传播
        if hops < max_hops:
            for neighbor in curr_node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
            

# 定义具体生物单元类
class PhotosynthesisUnit(OrganismUnit):
    """光合单元: 将光照转换为周围单元的能量"""
    __slots__ = ()
    def __repr__(self):
        return f"PhotosynthesisUnit(E={self.energy:.1f})"

    def update(self, node, all_nodes):
        # 基础代谢
        self.energy -= 2.0

        # 维护补给状态
        if self.supply_ttl > 0:
            self.supply_ttl -= 1
        else:
            self.supply_target = None

        neighbors = get_neighbors(node)
        
        # 防御机制: 与周围的敌对战斗单元一换一 (无视能量)
        my_species = self.dna.get('species_id', 0)
        for neighbor in neighbors:
            target_unit = neighbor.organism
            # 使用类名判断以避免定义顺序问题
            if target_unit and target_unit.__class__.__name__ == 'CombatUnit':
                if target_unit.dna.get('species_id', 0) != my_species:
                    kill_unit(neighbor) # 杀死敌人
                    kill_unit(node)   # 牺牲自己
                    return

        # 获取当前格子的光照
        light = node.light
        
        # 线性公式转换能量 (例如系数为 100)
        generated_energy = light * 100.0
        
        if not neighbors:
            return

        # 将能量分发给周围单元
        targets = []
        priority_targets = [] # 优先目标(前线)
        
        my_dist = float('inf')
        if self.supply_target:
             my_dist = (node.x - self.supply_target[0])**2 + (node.y - self.supply_target[1])**2

        for neighbor in neighbors:
            if neighbor.organism:
                target_org = neighbor.organism
                targets.append(target_org)
                # 如果有补给目标，优先给距离目标更近的单位
                if self.supply_target:
                    dist = (neighbor.x - self.supply_target[0])**2 + (neighbor.y - self.supply_target[1])**2
                    if dist < my_dist:
                        priority_targets.append(target_org)
        
        final_targets = priority_targets if priority_targets else targets
        
        if final_targets:
            share = generated_energy / len(final_targets)
            for target in final_targets:
                target.energy += share

class ReplicationUnit(OrganismUnit):
    """复制单元: 积累能量进行复制和变身"""
    __slots__ = ('replication_constant', 'accumulated_energy', 'failed_attempts')
    def __init__(self, dna=None):
        super().__init__(dna)
        self.replication_constant = self.dna.get('replication_threshold', 50.0) # 繁殖所需的能量阈值
        self.accumulated_energy = 0.0
        self.failed_attempts = 0

    def __repr__(self):
        return f"ReplicationUnit(E={self.energy:.1f}, Acc={self.accumulated_energy:.1f})"

    def respond_to_alert(self, node, enemy_pos):
        # 紧急防御: 如果能量充足，生成战斗单元
        if self.energy > 20.0:
            neighbors = get_neighbors(node)
            empty = [n for n in neighbors if n.organism is None]
            if empty:
                # 选择离敌人最近的空位
                best_spot = min(empty, key=lambda n: (n.x-enemy_pos[0])**2 + (n.y-enemy_pos[1])**2)
                
                self.energy -= 10.0
                defender = CombatUnit(dna=self.dna)
                defender.energy = 10.0
                defender.alert_target = enemy_pos # 告知敌人位置
                spawn_unit(best_spot, defender)

    def update(self, node, all_nodes):
        # 基础代谢
        self.energy -= 1.0

        # 战斗检测: 如果周围有异种生物，变身为战斗单元
        neighbors = get_neighbors(node)
        my_species = self.dna.get('species_id', 0)
        
        for neighbor in neighbors:
            neighbor_unit = neighbor.organism
            if neighbor_unit:
                neighbor_species = neighbor_unit.dna.get('species_id', 0)
                if neighbor_species != my_species:
                    # 发现敌人，变身
                    combat_unit = CombatUnit(dna=self.dna)
                    combat_unit.energy = self.energy
                    spawn_unit(node, combat_unit)
                    return

        # 退化逻辑: 有一定概率退化为连接单元
        degeneration_prob = self.dna.get('degeneration_prob', 0.0)
        if random.random() < degeneration_prob:
            new_unit = ConnectionUnit(dna=self.dna)
            new_unit.energy = self.energy # 继承能量
            spawn_unit(node, new_unit)
            return

        # 逻辑：以 常数/自身能量 的时间进行动作
        self.accumulated_energy += self.energy
        self.energy = 0 
        
        # 计算动态繁殖阈值: 基础阈值 - (连接数 * 敏感度 * 5.0)
        # 敏感度为正: 连接越多，繁殖越容易 (阈值降低)
        # 敏感度为负: 连接越多，繁殖越困难 (阈值升高)
        degree = len(get_neighbors(node))
        sensitivity = self.dna.get('replication_connection_sensitivity', 0.0)
        effective_threshold = self.replication_constant - (degree * sensitivity * 5.0)
        effective_threshold = max(10.0, effective_threshold) # 设定最低能量下限

        if self.accumulated_energy >= effective_threshold:
            if self.trigger_event(node):
                self.accumulated_energy = 0 # 重置

    def trigger_event(self, node):
        neighbors = get_neighbors(node)
        # 1. 在周围任意一格产生新的复制单元
        empty_neighbors = [n for n in neighbors if n.organism is None]
        
        if not empty_neighbors:
            # 无法复制
            self.failed_attempts += 1
            if self.failed_attempts >= 1:
                # 超过一回合无法复制，退化为连接单元
                spawn_unit(node, ConnectionUnit(dna=self.dna))
                return True
            return False

        # 可以复制
        target_node = random.choice(empty_neighbors)
        # 传递DNA给后代
        spawn_unit(target_node, ReplicationUnit(dna=self.dna))

        # 2. 将自己变为 光合单元 或 连接单元 或 储能单元
        # 根据DNA中的光照阈值决定
        current_light = node.light
        
        light_thresh = self.dna.get('light_threshold', 0.5)
        
        # --- 基于连接数和DNA策略的变身逻辑 ---
        valid_neighbors_count = len(neighbors)
        conn_thresh = self.dna.get('connection_threshold', 3.0)
        hub_behavior = self.dna.get('hub_behavior', 0.5)
        corridor_behavior = self.dna.get('corridor_behavior', 0.4)
        
        new_type = None
        
        if valid_neighbors_count >= conn_thresh:
            # 枢纽节点 (Hub)
            if hub_behavior > 0.7:
                # 倾向于储能 (能量银行)
                new_type = StorageUnit
            elif hub_behavior > 0.3:
                # 倾向于光合 (资源枢纽)
                new_type = PhotosynthesisUnit
            else:
                # 倾向于连接 (交通枢纽)
                new_type = ConnectionUnit
        else:
            # 通道节点 (Corridor)
            if corridor_behavior > 0.6:
                # 倾向于光合 (拾荒者)
                new_type = PhotosynthesisUnit
            else:
                # 倾向于连接 (快速通道)
                new_type = ConnectionUnit
        
        # --- 环境限制修正 ---
        if new_type == PhotosynthesisUnit:
            # 如果光照太差，无法维持光合，转为连接
            if not (current_light > light_thresh):
                new_type = ConnectionUnit
            
        spawn_unit(node, new_type(dna=self.dna))
        return True

class SporeUnit(OrganismUnit):
    """孢子单元: 携带DNA，一刻后爆发"""
    __slots__ = ()
    def __repr__(self):
        return "SporeUnit"

    def update(self, node, all_nodes):
        # 爆发逻辑: 变成一个光合单元与周围的复制单元
        
        # 1. 中心变为光合单元
        spawn_unit(node, PhotosynthesisUnit(dna=self.dna))
        
        # 2. 四周变为复制单元
        neighbors = get_neighbors(node)
        for neighbor in neighbors:
            spawn_unit(neighbor, ReplicationUnit(dna=self.dna))

class ConnectionUnit(OrganismUnit):
    """连接单元: 将能量传递给周围单元"""
    __slots__ = ()
    def __repr__(self):
        return f"ConnectionUnit(E={self.energy:.1f})"

    def update(self, node, all_nodes):
        # 基础代谢
        self.energy -= 1.0

        # 维护补给状态
        if self.supply_ttl > 0:
            self.supply_ttl -= 1
        else:
            self.supply_target = None

        # 变异逻辑: 有一定概率变异为复制单元
        mutation_prob = self.dna.get('mutation_prob', 0.0)
        if random.random() < mutation_prob:
            new_unit = ReplicationUnit(dna=self.dna)
            new_unit.energy = self.energy # 继承能量
            spawn_unit(node, new_unit)
            return

        if self.energy <= 0.1: # 能量太少不传输
            return

        neighbors = get_neighbors(node)
        if not neighbors:
            return

        # 寻找周围有生物的格子
        targets = []
        priority_targets = [] # 优先目标(前线)
        storage_targets = [] # 储能目标
        
        my_dist = float('inf')
        if self.supply_target:
             my_dist = (node.x - self.supply_target[0])**2 + (node.y - self.supply_target[1])**2

        for neighbor in neighbors:
            target_org = neighbor.organism
            if target_org:
                # 识别储能单元
                if isinstance(target_org, StorageUnit):
                    storage_targets.append(target_org)
                else:
                    targets.append(target_org)
                
                # 如果有补给目标，优先给距离目标更近的单位
                if self.supply_target:
                    dist = (neighbor.x - self.supply_target[0])**2 + (neighbor.y - self.supply_target[1])**2
                    if dist < my_dist:
                        priority_targets.append(target_org)
        
        # 优先级: 优先目标 > 普通目标 > 储能单元 (只有当没有其他需求时才存起来)
        # 或者: 策略性存储? 
        # 简单策略: 如果能量非常充裕 (>50), 分一部分给储能; 否则优先给普通单位
        
        final_targets = []
        if priority_targets:
            final_targets = priority_targets
        elif targets:
            final_targets = targets
        elif storage_targets:
            final_targets = storage_targets
            
        if final_targets:
            # 传输自身大部分能量，例如 80%
            transfer_total = self.energy * 0.8
            self.energy -= transfer_total
            
            # 平均分配给所有邻居
            share = transfer_total / len(final_targets)
            received_amount = share * 1
            
            for org in final_targets:
                org.energy += received_amount

class CombatUnit(OrganismUnit):
    """战斗单元: 攻击异种生物并同化"""
    __slots__ = ('patience', 'alert_target')
    def __init__(self, dna=None):
        super().__init__(dna)
        self.patience = 5 # 战斗状态维持回合数 (滞留时间)
        self.alert_target = None # 警报目标 (x, y)

    def __repr__(self):
        return f"CombatUnit(E={self.energy:.1f})"

    def respond_to_alert(self, node, enemy_pos):
        # 收到警报，设定目标并重置耐心
        self.alert_target = enemy_pos
        self.patience = 10

    def update(self, node, all_nodes):
        # 基础代谢 (战斗单元消耗更高，维持军队需要代价)
        self.energy -= 3.0
        if self.energy <= 0:
            # 死亡由主循环处理，但这里可以提前返回
            return

        neighbors = get_neighbors(node)
        my_species = self.dna.get('species_id', 0)
        has_enemy = False
        
        # 1. 优先攻击周围的敌人
        for neighbor in neighbors:
            target_unit = neighbor.organism
            if target_unit:
                target_species = target_unit.dna.get('species_id', 0)
                if target_species != my_species:
                    has_enemy = True
                    
                    # 发出警报 (呼叫支援)
                    alert_allies(all_nodes, node, my_species, (neighbor.x, neighbor.y))
                    # 敌人也发出警报
                    alert_allies(all_nodes, neighbor, target_species, (node.x, node.y))
                    
                    # 战斗逻辑优化: 伤害交换机制
                    # 双方同时造成伤害，伤害量与自身能量相关
                    my_damage = self.energy * 0.5
                    enemy_damage = target_unit.energy * 0.5
                    
                    # 造成伤害
                    target_unit.energy -= my_damage
                    self.energy -= enemy_damage * 0.5 # 攻击者受到的反击伤害较小 (优势)
                    
                    # 判定结果
                    if target_unit.energy <= 0:
                        # 敌人死亡，尝试占领/同化
                        # 如果自己还活着且能量足够
                        if self.energy > 10.0:
                            self.energy -= 10.0
                            # 在敌人位置生成己方复制单元
                            new_unit = ReplicationUnit(dna=self.dna)
                            new_unit.energy = 10.0
                            spawn_unit(neighbor, new_unit)
                        else:
                            # 仅杀死，不占领
                            kill_unit(neighbor)
                    
                    if self.energy <= 0:
                        kill_unit(node)
                        return # 自己死亡，结束
        
        # 状态维护逻辑
        if has_enemy:
            self.patience = 5 # 发现敌人，重置倒计时
            self.alert_target = None # 既然已经在战斗，清除远程目标
        else:
            self.patience -= 1
            
            # 主动移动逻辑 (巡逻/猎杀/响应警报)
            # 如果能量充足，尝试移动
            if self.energy > 5.0:
                move_to = None
                
                # 1. 响应警报优先
                if self.alert_target:
                    # 检查目标是否已经到达或无效
                    if (node.x, node.y) == self.alert_target:
                        self.alert_target = None
                    else:
                        # 简单的贪婪寻路向警报点移动
                        empty_neighbors = [n for n in neighbors if n.organism is None]
                        if empty_neighbors:
                            best_spot = min(empty_neighbors, key=lambda n: (n.x-self.alert_target[0])**2 + (n.y-self.alert_target[1])**2)
                            move_to = best_spot
                            
                            # 如果距离很近了，可能已经到达附近，清除目标以防卡死
                            if (node.x-self.alert_target[0])**2 + (node.y-self.alert_target[1])**2 < 2:
                                self.alert_target = None

                # 2. 猎杀模式 (如果没有警报目标)
                if not move_to:
                    # 猎杀模式: 扫描半径2的范围寻找敌人 (Graph mode: neighbors of neighbors)
                    target_pos = None
                    
                    # BFS depth 2
                    visited = {node}
                    queue = [(node, 0)]
                    found_enemy = False
                    
                    while queue:
                        curr, depth = queue.pop(0)
                        if depth >= 2: continue
                        
                        for n in curr.neighbors:
                            if n not in visited:
                                visited.add(n)
                                u = n.organism
                                if u and u.dna.get('species_id', 0) != my_species:
                                    target_pos = (n.x, n.y)
                                    found_enemy = True
                                    break
                                queue.append((n, depth + 1))
                        if found_enemy: break
                    
                    empty_neighbors = [n for n in neighbors if n.organism is None]
                    
                    if target_pos and empty_neighbors:
                        # 向敌人方向移动
                        best_dist = 999
                        for n in empty_neighbors:
                            dist = ((n.x - target_pos[0])**2 + (n.y - target_pos[1])**2)
                            if dist < best_dist:
                                best_dist = dist
                                move_to = n
                    elif empty_neighbors:
                        # 随机巡逻
                        move_to = random.choice(empty_neighbors)
                
                if move_to:
                    me = self
                    kill_unit(node) 
                    spawn_unit(move_to, me)
                    return

        # 如果倒计时结束且无敌人，变回复制单元以继续发展
        if self.patience <= 0:
            new_unit = ReplicationUnit(dna=self.dna)
            new_unit.energy = self.energy
            spawn_unit(node, new_unit)

class StorageUnit(OrganismUnit):
    """储能单元: 高容量电池，用于存储能量"""
    __slots__ = ()
    def __init__(self, dna=None):
        super().__init__(dna)
        self.energy = 50.0 # 初始拥有较多能量

    def __repr__(self):
        return f"StorageUnit(E={self.energy:.1f})"

    def update(self, node, all_nodes):
        # 极低的基础代谢，适合长期存储
        self.energy -= 0.1
        
        # 如果能量过高，可能会"泄漏"给周围贫瘠的邻居 (被动溢出)
        if self.energy > 200.0:
            neighbors = get_neighbors(node)
            for n in neighbors:
                if n.organism and n.organism.energy < 50.0:
                    transfer = 10.0
                    self.energy -= transfer
                    n.organism.energy += transfer

class FactoryUnit(OrganismUnit):
    """工厂单元: (已废弃)"""
    __slots__ = ()
    def __repr__(self):
        return f"FactoryUnit(E={self.energy:.1f})"

    def update(self, node, all_nodes):
        # 消耗能量
        consumption = 5.0
        if self.energy < consumption:
            # 能量不足，无法工作，甚至可能死亡(由主循环处理)
            return
            
        self.energy -= consumption
        
        # 肥力机制已移除，工厂单元不再产生效果

# 定义节点类
class Node:
    __slots__ = ('id', 'x', 'y', 'neighbors', 'light', 'base_light', 'organism')
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.neighbors = []
        # 光照（浮点数，范围0~1）
        self.base_light = random.random()
        self.light = self.base_light
        # 其上生物（默认为无）
        self.organism = None

    def __repr__(self):
        org_str = self.organism if self.organism else "None"
        return f"Node({self.id}, Org={org_str})"

def spawn_unit(node, unit, apply_fertility=True):
    node.organism = unit
    
    # 光照惩罚: 光合单元周围节点减半
    if isinstance(unit, PhotosynthesisUnit):
        # 简单起见，只影响直接邻居
        for n in node.neighbors:
            n.light *= 0.5

def kill_unit(node):
    """移除生物并恢复环境影响"""
    unit = node.organism
    if unit is None:
        return
    
    # 光照恢复: 光合单元周围节点恢复
    if isinstance(unit, PhotosynthesisUnit):
        for n in node.neighbors:
            n.light *= 2.0
    
    node.organism = None



def generate_graph_world(num_nodes, width, height):
    """生成图结构世界"""
    print(f"正在生成 {num_nodes} 个节点的图结构世界...")
    nodes = []
    
    # 1. 生成随机节点
    for i in range(num_nodes):
        # 留出边距
        x = random.randint(5, height-6)
        y = random.randint(5, width-6)
        nodes.append(Node(i, x, y))
        
    # 2. 连接节点 (MST + K近邻)
    # 优化: 使用网格空间划分加速距离计算
    edges = set()
    
    # 3.0 空间划分
    grid_size = 20 # 网格大小
    grid = {}
    for node in nodes:
        gx = int(node.x // grid_size)
        gy = int(node.y // grid_size)
        key = (gx, gy)
        if key not in grid: grid[key] = []
        grid[key].append(node)

    # 3.1 准备边列表 (只计算相邻网格的节点)
    all_possible_edges = []
    
    # 遍历每个网格
    for (gx, gy), cell_nodes in grid.items():
        # 遍历当前网格及周围网格
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (gx + dx, gy + dy)
                if neighbor_key in grid:
                    other_nodes = grid[neighbor_key]
                    
                    # 计算距离
                    for n1 in cell_nodes:
                        for n2 in other_nodes:
                            if n1.id < n2.id: # 避免重复和自环
                                d = (n1.x-n2.x)**2 + (n1.y-n2.y)**2
                                # 只保留一定范围内的边用于MST，避免过长边
                                if d < (grid_size * 3)**2: 
                                    all_possible_edges.append((d, n1.id, n2.id))
    
    all_possible_edges.sort()
    
    # 3.2 Kruskal算法生成最小生成树
    parent = list(range(len(nodes)))
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False
    
    # MST 连接
    mst_edges_count = 0
    for _, u, v in all_possible_edges:
        if union(u, v):
            edges.add(tuple(sorted((u, v))))
            nodes[u].neighbors.append(nodes[v])
            nodes[v].neighbors.append(nodes[u])
            mst_edges_count += 1
            
    # 确保图是连通的 (如果MST没连通所有点，说明有些点太远了，强制连接最近的点)
    # 简单的补救措施：检查连通分量
    root_set = set(find(i) for i in range(len(nodes)))
    if len(root_set) > 1:
        print(f"警告: 图未完全连通，有 {len(root_set)} 个连通分量。尝试强制连接...")
        # 这里可以添加额外的逻辑来连接断开的部分，或者忽略
        # 简单做法：让所有孤立分量的根节点连接到节点0
        base_root = find(0)
        for i in range(1, len(nodes)):
            root_i = find(i)
            if root_i != base_root:
                # 强制连接 i 和 0 (虽然距离可能很远)
                edges.add(tuple(sorted((0, i))))
                nodes[0].neighbors.append(nodes[i])
                nodes[i].neighbors.append(nodes[0])
                union(0, i)

    # 3.3 叠加K近邻连接 (增加局部环路)
    # 同样利用网格加速查找
    for node in nodes:
        gx = int(node.x // grid_size)
        gy = int(node.y // grid_size)
        
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    candidates.extend(grid[key])
        
        distances = []
        for other in candidates:
            if node.id == other.id: continue
            d = (node.x-other.x)**2 + (node.y-other.y)**2
            distances.append((d, other))
        
        distances.sort(key=lambda x: x[0])
        
        # 连接最近的5个点
        for k in range(min(5, len(distances))):
            other = distances[k][1]
            edge = tuple(sorted((node.id, other.id)))
            if edge not in edges:
                edges.add(edge)
                node.neighbors.append(other)
                other.neighbors.append(node)
    
    # 设置资源分布 (均匀随机)
    for node in nodes:
        node.base_light = random.random()
        node.light = node.base_light

    print("世界生成完毕。")
    return nodes, list(edges)

def save_top_species(all_nodes, filename="top_species.json"):
    print("正在保存前10名物种DNA...")
    species_counts = {}
    species_dnas = {}
    
    for node in all_nodes:
        unit = node.organism
        if unit:
            dna = unit.dna
            s_id = dna.get('species_id')
            if s_id is not None:
                species_counts[s_id] = species_counts.get(s_id, 0) + 1
                if s_id not in species_dnas:
                    species_dnas[s_id] = dna
    
    sorted_species = sorted(species_counts.items(), key=lambda item: item[1], reverse=True)
    top_10 = sorted_species[:10]
    
    # 保存格式: [{'dna': ..., 'name': ...}, ...]
    saved_data = []
    for s_id, count in top_10:
        saved_data.append({
            'dna': species_dnas[s_id],
            'name': get_species_name(s_id)
        })
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4, ensure_ascii=False)
        print(f"已保存 {len(saved_data)} 个物种DNA及名字到 {filename}")
    except Exception as e:
        print(f"保存失败: {e}")

def load_top_species(filename="top_species.json"):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dnas = []
        if isinstance(data, list):
            for item in data:
                # 兼容旧格式 (直接是DNA字典) 和新格式 (包含name的字典)
                if 'dna' in item and 'name' in item:
                    dna = item['dna']
                    name = item['name']
                    s_id = dna.get('species_id')
                    if s_id is not None:
                        SPECIES_NAMES[s_id] = name
                    dnas.append(dna)
                else:
                    dnas.append(item)
                    
        print(f"从 {filename} 读取了 {len(dnas)} 个物种DNA")
        return dnas
    except Exception as e:
        print(f"读取失败: {e}")
        return None

def populate_randomly(all_nodes, count=100, initial_dnas=None):
    """随机放置一些生物单元用于演示"""
    
    # 辅助函数: 寻找有效位置
    def get_valid_node():
        for _ in range(50):
            node = random.choice(all_nodes)
            if node.organism is None:
                return node
        return None

    if initial_dnas:
        print(f"使用存档的 {len(initial_dnas)} 个DNA初始化...")
        for dna in initial_dnas:
             node = get_valid_node()
             if node:
                 spawn_unit(node, SporeUnit(dna=dna))
        
        # 如果不足10个，补齐随机
        remaining = 10 - len(initial_dnas)
        if remaining > 0:
             for _ in range(remaining):
                node = get_valid_node()
                if node:
                    dna = {
                        'light_threshold': random.uniform(0.2, 0.8),
                        'fertility_threshold': random.uniform(0.2, 0.8),
                        'replication_threshold': random.uniform(20.0, 80.0),
                        'mutation_prob': random.uniform(0.0, 0.05),
                        'degeneration_prob': random.uniform(0.0, 0.05),
                        'fertility_sacrifice_threshold': random.uniform(0.0, 0.3),
                        'factory_prob': random.uniform(0.0, 0.1),
                        'factory_fertility_threshold': random.uniform(0.1, 0.5),
                        'species_id': random.randint(1, 100)
                    }
                    spawn_unit(node, SporeUnit(dna=dna))
    else:
        # 放置一些孢子单元
        for _ in range(10):
            node = get_valid_node()
            if node:
                # 随机DNA阈值
                dna = {
                    'light_threshold': random.uniform(0.2, 0.8),
                    'fertility_threshold': random.uniform(0.2, 0.8),
                    'replication_threshold': random.uniform(20.0, 80.0),
                    'mutation_prob': random.uniform(0.0, 0.05), # 0% ~ 5%
                    'degeneration_prob': random.uniform(0.0, 0.05), # 0% ~ 5%
                    'fertility_sacrifice_threshold': random.uniform(0.0, 0.3),
                    'factory_prob': random.uniform(0.0, 0.1),
                    'factory_fertility_threshold': random.uniform(0.1, 0.5),
                    'connection_threshold': random.uniform(2.0, 6.0),
                    'replication_connection_sensitivity': random.uniform(-0.5, 0.5),
                    'hub_behavior': random.uniform(0.0, 1.0),
                    'corridor_behavior': random.uniform(0.0, 1.0),
                    'species_id': random.randint(1, 100)
                }
                spawn_unit(node, SporeUnit(dna=dna))

    # 移除普通单元的生成，只保留孢子
    # unit_types = [PhotosynthesisUnit, ReplicationUnit, ConnectionUnit]
    # for _ in range(count):
    #     ...

def mutate_dna(dna):
    """对DNA进行轻微变异"""
    new_dna = dna.copy()
    for key, value in new_dna.items():
        if isinstance(value, float):
            # 变异幅度 +/- 10%
            change = value * random.uniform(-0.1, 0.1)
            new_value = value + change
            
            # 限制范围
            if 'prob' in key: # 概率 0-1
                new_value = max(0.0, min(1.0, new_value))
            elif 'sensitivity' in key: # 敏感度 -1 ~ 1
                new_value = max(-1.0, min(1.0, new_value))
            elif 'threshold' in key:
                if 'replication' in key: # 能量阈值
                    new_value = max(10.0, new_value)
                else: # 其他阈值 0-1
                    new_value = max(0.0, min(1.0, new_value))
            
            new_dna[key] = new_value
    return new_dna

def perform_pollination(all_nodes):
    # 1. 随机选取源位置
    source_unit = None
    for _ in range(20):
        node = random.choice(all_nodes)
        if node.organism:
            source_unit = node.organism
            break
    
    if source_unit:
        base_dna = source_unit.dna
    else:
        # 无生物则随机生成DNA
        base_dna = {
            'light_threshold': random.uniform(0.2, 0.8),
            'fertility_threshold': random.uniform(0.2, 0.8),
            'replication_threshold': random.uniform(20.0, 80.0),
            'mutation_prob': random.uniform(0.0, 0.05),
            'degeneration_prob': random.uniform(0.0, 0.05),
            'fertility_sacrifice_threshold': random.uniform(0.0, 0.3),
            'factory_prob': random.uniform(0.0, 0.1),
            'factory_fertility_threshold': random.uniform(0.1, 0.5),
            'connection_threshold': random.uniform(2.0, 6.0),
            'replication_connection_sensitivity': random.uniform(-0.5, 0.5),
            'hub_behavior': random.uniform(0.0, 1.0),
            'corridor_behavior': random.uniform(0.0, 1.0),
            'species_id': random.randint(1, 100)
        }
        
    # 2. 寻找目标空位
    for _ in range(20):
        node = random.choice(all_nodes)
        if node.organism is None:
            # 3. 生成变异孢子
            new_dna = mutate_dna(base_dna)
            spawn_unit(node, SporeUnit(dna=new_dna))
            break

def simulate_step(all_nodes, step_count=0):
    """执行一步模拟"""
    
    # --- 环境季节循环 ---
    # 周期为 200 步
    # 光照在 0.0 ~ 2.0 倍之间波动
    season_factor = 1.0 + 1 * np.sin(step_count * 2 * np.pi / 200)
    
    # 更新所有节点的光照 (基础随机值 * 季节因子)
    # 注意: 这里我们假设 node.light 存储的是当前值，我们需要一个 base_light 来存储基础值
    # 但为了节省内存不加新属性，我们直接修改 light，但这会导致漂移。
    # 正确做法: 既然 light 已经是随机的，我们就在 PhotosynthesisUnit 计算能量时乘上季节因子
    # 或者，我们在这里统一更新，但需要保持基础值。
    # 简单方案: 每次重新生成随机基础值? 不行。
    # 方案: 让 PhotosynthesisUnit 读取全局季节因子。
    # 为了简单，我们将 season_factor 注入到 global 或作为参数传递?
    # 这里我们选择直接修改 node.light，但需要一种方式恢复。
    # 鉴于 Node 结构，我们添加一个 base_light 属性比较好，或者利用 id 伪随机生成 base_light。
    
    # 采用方案: 在 PhotosynthesisUnit 中计算。
    # 为了可视化能看到光照变化，我们需要修改 node.light。
    # 让我们给 Node 加一个 base_light 属性。
    
    # 传粉机制 (每10步)
    if step_count % 10 == 0:
        perform_pollination(all_nodes)
    
    # 1. 更新所有节点的基础光照 (模拟季节)
    for node in all_nodes:
        if hasattr(node, 'base_light'):
            node.light = node.base_light * max(0.0, season_factor)

    # 2. 应用光合单元的遮蔽效应 (Shading Effect)
    # 光合单元会遮挡周围节点的光照
    shading_map = {} # node -> shading_factor
    
    for node in all_nodes:
        if isinstance(node.organism, PhotosynthesisUnit):
            neighbors = get_neighbors(node)
            for neighbor in neighbors:
                # 累加遮挡因子 (例如每个光合单元遮挡 80%)
                shading_map[neighbor] = shading_map.get(neighbor, 1.0) * 0.2
    
    # 应用遮挡
    for node, factor in shading_map.items():
        node.light *= factor

    # 3. 遍历所有节点进行生物更新
    for node in all_nodes:
        unit = node.organism
        if unit:
            # 能量检查: 能量 < 0 死亡
            if unit.energy < 0:
                kill_unit(node)
                continue
            
            unit.update(node, all_nodes)

def run_batch_simulation(rounds=10, steps=1000, width=100, height=100):
    """运行无可视化的批量模拟"""
    print(f"启动批量模拟模式: 共 {rounds} 轮, 每轮 {steps} 帧")
    
    for r in range(1, rounds + 1):
        print(f"\n--- 第 {r} / {rounds} 轮 ---")
        # 1. 生成世界
        nodes, edges = generate_graph_world(400, width, height)
        
        # 2. 读取存档 (上一轮的优胜者)
        saved_dnas = load_top_species()
        
        # 3. 初始生物
        populate_randomly(nodes, count=300, initial_dnas=saved_dnas)
        
        # 4. 模拟循环
        print("正在模拟...")
        for s in range(steps):
            simulate_step(nodes, step_count=s)
            if (s + 1) % 100 == 0:
                print(f"  进度: {s + 1} / {steps}")
        
        # 5. 保存结果
        save_top_species(nodes)
        print(f"第 {r} 轮完成，已保存优势物种存档。")
        
    print("\n所有模拟轮次结束。")

if __name__ == "__main__":
    # 检查命令行参数是否为 batch
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        run_batch_simulation()
        sys.exit()

    WIDTH = 100 
    HEIGHT = 100
    
    # 1. 生成世界
    nodes, edges = generate_graph_world(1000, WIDTH, HEIGHT)
    
    # 读取存档
    saved_dnas = load_top_species()
    
    # 2. 随机放置一些生物进行演示
    populate_randomly(nodes, count=300, initial_dnas=saved_dnas)
    
    print("初始化可视化窗口...")
    
    # 设置图形: 2x2 子图 (移除肥力图)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2 = axs[0]
    ax3, ax5 = axs[1]
    
    # 标题设置
    ax1.set_title("Organism Types")
    ax2.set_title("DNA (R:Light, G:Rep, B:Unused)")
    ax5.set_title("Energy Levels (Yellow)")
    ax3.set_title("Light Distribution")
    
    # 统一设置坐标轴范围和隐藏坐标轴
    plot_axes = [ax1, ax2, ax3, ax5]
    for ax in plot_axes:
        ax.set_xlim(0, HEIGHT)
        ax.set_ylim(0, WIDTH)
        ax.axis('off')

    # 预计算边的数据 (所有图层共享结构)
    from matplotlib.collections import LineCollection
    lines = []
    for u_idx, v_idx in edges:
        p1 = (nodes[u_idx].x, nodes[u_idx].y)
        p2 = (nodes[v_idx].x, nodes[v_idx].y)
        lines.append([p1, p2])
    
    # 为每个图层添加边和节点散点
    scatters = []
    
    # 辅助函数：初始化图层
    def init_layer(ax, edge_color='gray', edge_alpha=0.3):
        lc = LineCollection(lines, colors=edge_color, linewidths=0.5, alpha=edge_alpha)
        ax.add_collection(lc)
        x_vals = [n.x for n in nodes]
        y_vals = [n.y for n in nodes]
        return ax.scatter(x_vals, y_vals, s=50)

    scat1 = init_layer(ax1) # Types
    scat2 = init_layer(ax2) # DNA
    scat3 = init_layer(ax3) # Light
    scat5 = init_layer(ax5) # Energy

    # 排行榜文本 (放在图外或覆盖在某个图上，这里放在ax3左上角)
    ranking_text = ax3.text(0.05, 0.95, "", transform=ax3.transAxes, va='top', fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 季节显示文本
    season_text = ax3.text(0.05, 0.05, "", transform=ax3.transAxes, va='bottom', fontsize=12, fontweight='bold', color='yellow', bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    def update_plot(frame):
        simulate_step(nodes, step_count=frame)
        
        # 准备颜色数组
        c_types = []
        c_dna = []
        c_light = []
        c_energy = []
        
        for n in nodes:
            # 1. Organism Types
            if n.organism:
                unit = n.organism
                if isinstance(unit, PhotosynthesisUnit):
                    c_types.append([0, 1, 0])
                elif isinstance(unit, ReplicationUnit):
                    c_types.append([1, 0, 0])
                elif isinstance(unit, ConnectionUnit):
                    c_types.append([0, 0, 1])
                elif isinstance(unit, StorageUnit):
                    c_types.append([0, 1, 1]) # 青色: 储能单元
                elif isinstance(unit, FactoryUnit):
                    c_types.append([1, 0, 1])
                elif isinstance(unit, CombatUnit):
                    c_types.append([1, 0.5, 0])
                else:
                    c_types.append([0.5, 0.5, 0.5])
            else:
                c_types.append([0.1, 0.1, 0.1]) # 空节点
            
            # 2. DNA
            if n.organism:
                dna = n.organism.dna
                r = dna.get('light_threshold', 0.0)
                g = (dna.get('replication_threshold', 50.0) - 20.0) / 60.0
                g = max(0.0, min(1.0, g))
                b = 0.0 # 肥力已移除
                c_dna.append([r, g, b])
            else:
                c_dna.append([0.1, 0.1, 0.1])

            # 3. Light
            l = n.light
            c_light.append([l, l, l])

            # 5. Energy
            if n.organism:
                e = min(1.0, max(0.0, n.organism.energy / 100.0))
                c_energy.append([e, e, 0])
            else:
                c_energy.append([0.1, 0.1, 0.1])

        # 更新散点颜色
        scat1.set_color(c_types)
        scat2.set_color(c_dna)
        scat3.set_color(c_light)
        scat5.set_color(c_energy)
        
        # 更新排行榜
        species_counts = {}
        for n in nodes:
            if n.organism:
                s_id = n.organism.dna.get('species_id', 0)
                species_counts[s_id] = species_counts.get(s_id, 0) + 1
        
        sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        txt = "Top Species:\n"
        for s_id, count in sorted_species:
            name = get_species_name(s_id)
            txt += f"{name}: {count}\n"
        ranking_text.set_text(txt)
        
        # 更新季节显示
        season_val = np.sin(frame * 2 * np.pi / 200)
        season_name = "Spring/Autumn"
        if season_val > 0.5: season_name = "Summer (High Light)"
        elif season_val < -0.5: season_name = "Winter (Low Light)"
        
        light_factor = 1.0 + 0.5 * season_val
        season_text.set_text(f"Season: {season_name}\nLight Factor: {light_factor:.2f}")
        
        return scat1, scat2, scat3, scat5, ranking_text, season_text

    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)
    
    # 交互控制
    is_paused = False
    
    def on_key(event):
        global is_paused
        if event.key == ' ':
            if is_paused:
                ani.event_source.start()
                is_paused = False
                fig.suptitle("Ecosystem Graph Simulation")
            else:
                ani.event_source.stop()
                is_paused = True
                fig.suptitle("Ecosystem Graph Simulation - Paused")
            fig.canvas.draw_idle()
        elif event.key == 'q' or event.key == 'escape':
            save_top_species(nodes)
            plt.close(fig)
            print("模拟结束")

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("开始模拟。请查看弹出的窗口。")
    print("控制说明: 按 [空格] 暂停/继续，按 [q] 或 [Esc] 退出。")
    plt.show()

