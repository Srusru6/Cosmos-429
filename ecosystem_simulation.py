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
            'hub_behavior': 0.5, # 枢纽策略: <0.3连接, 0.3-0.7光合, >0.7工厂
            'corridor_behavior': 0.4, # 通道策略: <0.6连接, >0.6光合
            'species_id': 0 # 种群ID
        }

    def update(self, board, x, y):
        # 基础代谢消耗
        self.energy -= 0.5
        pass

    def respond_to_alert(self, board, x, y, enemy_pos):
        """响应警报: 默认无反应"""
        pass

def get_neighbors(board, x, y):
    """获取周围8个格子的坐标"""
    height = len(board)
    width = len(board[0])
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx < height and 0 <= ny < width:
                # 只有非虚空区域才是有效邻居
                if not board[nx][ny].is_void:
                    neighbors.append((nx, ny))
    return neighbors

def alert_allies(board, center_x, center_y, species_id, enemy_pos, radius=6):
    """警报机制: 通知周围同种生物有敌人"""
    height = len(board)
    width = len(board[0])
    
    r_min = max(0, center_x - radius)
    r_max = min(height, center_x + radius + 1)
    c_min = max(0, center_y - radius)
    c_max = min(width, center_y + radius + 1)
    
    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            unit = board[r][c].organism
            if unit and unit.dna.get('species_id') == species_id:
                unit.respond_to_alert(board, r, c, enemy_pos)
            

# 定义具体生物单元类
class PhotosynthesisUnit(OrganismUnit):
    """光合单元: 将光照转换为周围单元的能量"""
    def __repr__(self):
        return f"PhotosynthesisUnit(E={self.energy:.1f})"

    def update(self, board, x, y):
        neighbors = get_neighbors(board, x, y)
        
        # 防御机制: 与周围的敌对战斗单元一换一 (无视能量)
        my_species = self.dna.get('species_id', 0)
        for nx, ny in neighbors:
            target_unit = board[nx][ny].organism
            # 使用类名判断以避免定义顺序问题
            if target_unit and target_unit.__class__.__name__ == 'CombatUnit':
                if target_unit.dna.get('species_id', 0) != my_species:
                    kill_unit(board, nx, ny) # 杀死敌人
                    kill_unit(board, x, y)   # 牺牲自己
                    return

        # 获取当前格子的光照
        light = board[x][y].light
        
        # 线性公式转换能量 (例如系数为 100)
        generated_energy = light * 100.0
        
        if not neighbors:
            return

        # 将能量分发给周围单元
        # 假设均匀分配给周围存在的生物
        targets = []
        for nx, ny in neighbors:
            if board[nx][ny].organism:
                targets.append(board[nx][ny].organism)
        
        if targets:
            share = generated_energy / len(targets)
            for target in targets:
                target.energy += share

class ReplicationUnit(OrganismUnit):
    """复制单元: 积累能量进行复制和变身"""
    def __init__(self, dna=None):
        super().__init__(dna)
        self.replication_constant = self.dna.get('replication_threshold', 50.0) # 繁殖所需的能量阈值
        self.accumulated_energy = 0.0
        self.failed_attempts = 0

    def __repr__(self):
        return f"ReplicationUnit(E={self.energy:.1f}, Acc={self.accumulated_energy:.1f})"

    def respond_to_alert(self, board, x, y, enemy_pos):
        # 紧急防御: 如果能量充足，生成战斗单元
        if self.energy > 20.0:
            neighbors = get_neighbors(board, x, y)
            empty = [n for n in neighbors if board[n[0]][n[1]].organism is None]
            if empty:
                # 选择离敌人最近的空位
                best_spot = min(empty, key=lambda p: (p[0]-enemy_pos[0])**2 + (p[1]-enemy_pos[1])**2)
                
                self.energy -= 10.0
                defender = CombatUnit(dna=self.dna)
                defender.energy = 10.0
                defender.alert_target = enemy_pos # 告知敌人位置
                spawn_unit(board, best_spot[0], best_spot[1], defender)

    def update(self, board, x, y):
        # 战斗检测: 如果周围有异种生物，变身为战斗单元
        neighbors = get_neighbors(board, x, y)
        my_species = self.dna.get('species_id', 0)
        
        for nx, ny in neighbors:
            neighbor_unit = board[nx][ny].organism
            if neighbor_unit:
                neighbor_species = neighbor_unit.dna.get('species_id', 0)
                if neighbor_species != my_species:
                    # 发现敌人，变身
                    combat_unit = CombatUnit(dna=self.dna)
                    combat_unit.energy = self.energy
                    spawn_unit(board, x, y, combat_unit)
                    return

        # 退化逻辑: 有一定概率退化为连接单元
        degeneration_prob = self.dna.get('degeneration_prob', 0.0)
        if random.random() < degeneration_prob:
            new_unit = ConnectionUnit(dna=self.dna)
            new_unit.energy = self.energy # 继承能量
            spawn_unit(board, x, y, new_unit)
            return

        # 逻辑：以 常数/自身能量 的时间进行动作
        self.accumulated_energy += self.energy
        self.energy = 0 
        
        if self.accumulated_energy >= self.replication_constant:
            if self.trigger_event(board, x, y):
                self.accumulated_energy = 0 # 重置

    def trigger_event(self, board, x, y):
        neighbors = get_neighbors(board, x, y)
        # 1. 在周围4格（上下左右）任意一格产生新的复制单元
        height = len(board)
        width = len(board[0])
        four_neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                four_neighbors.append((nx, ny))

        empty_neighbors = [n for n in four_neighbors if board[n[0]][n[1]].organism is None]
        
        if not empty_neighbors:
            # 无法复制
            self.failed_attempts += 1
            if self.failed_attempts >= 1:
                # 超过一回合无法复制，退化为连接单元
                spawn_unit(board, x, y, ConnectionUnit(dna=self.dna))
                return True
            return False

        # 可以复制
        nx, ny = random.choice(empty_neighbors)
        # 传递DNA给后代
        spawn_unit(board, nx, ny, ReplicationUnit(dna=self.dna))

        # 2. 将自己变为 光合单元 或 连接单元 或 工厂单元
        # 根据DNA中的光照阈值和肥力阈值决定
        current_light = board[x][y].light
        current_fertility = board[x][y].fertility
        
        light_thresh = self.dna.get('light_threshold', 0.5)
        fert_thresh = self.dna.get('fertility_threshold', 0.5)
        
        # 工厂单元判断逻辑
        factory_prob = self.dna.get('factory_prob', 0.05)
        factory_fert_thresh = self.dna.get('factory_fertility_threshold', 0.3)
        
        # 计算周围9格(包括自己)的平均肥力，以检测区域贫瘠程度
        total_fert = current_fertility
        count = 1
        for nx, ny in neighbors:
            total_fert += board[nx][ny].fertility
            count += 1
        avg_fertility = total_fert / count
        
        # --- 基于连接数和DNA策略的变身逻辑 ---
        valid_neighbors_count = len(neighbors)
        conn_thresh = self.dna.get('connection_threshold', 3.0)
        hub_behavior = self.dna.get('hub_behavior', 0.5)
        corridor_behavior = self.dna.get('corridor_behavior', 0.4)
        
        new_type = None
        
        if valid_neighbors_count >= conn_thresh:
            # 枢纽节点 (Hub)
            if hub_behavior > 0.7:
                # 倾向于工厂 (工业枢纽)
                new_type = FactoryUnit
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
        # 即使策略倾向于某种类型，如果环境条件极端不匹配，则退化
        
        if new_type == FactoryUnit:
            # 如果肥力其实很高，不需要工厂，转为光合
            if avg_fertility > factory_fert_thresh:
                new_type = PhotosynthesisUnit
        
        if new_type == PhotosynthesisUnit:
            # 如果光照或肥力太差，无法维持光合，转为连接
            if not (current_light > light_thresh and current_fertility > fert_thresh):
                new_type = ConnectionUnit
            
        spawn_unit(board, x, y, new_type(dna=self.dna))
        return True

class SporeUnit(OrganismUnit):
    """孢子单元: 携带DNA，一刻后爆发"""
    def __repr__(self):
        return "SporeUnit"

    def update(self, board, x, y):
        # 爆发逻辑: 变成一个光合单元与周围的四个复制单元
        height = len(board)
        width = len(board[0])
        
        # 1. 中心变为光合单元
        spawn_unit(board, x, y, PhotosynthesisUnit(dna=self.dna))
        
        # 2. 四周变为复制单元 (上下左右)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                # 强制覆盖，确保孢子展开
                spawn_unit(board, nx, ny, ReplicationUnit(dna=self.dna))

class ConnectionUnit(OrganismUnit):
    """连接单元: 将能量传递给周围单元"""
    def __repr__(self):
        return f"ConnectionUnit(E={self.energy:.1f})"

    def update(self, board, x, y):
        # 获取当前肥力
        current_fertility = board[x][y].fertility
        
        # 肥力特别小时牺牲自己
        fert_sacrifice_thresh = self.dna.get('fertility_sacrifice_threshold', 0.1)
        if current_fertility < fert_sacrifice_thresh:
            kill_unit(board, x, y)
            return

        # 变异逻辑: 有一定概率变异为复制单元
        mutation_prob = self.dna.get('mutation_prob', 0.0)
        if random.random() < mutation_prob:
            new_unit = ReplicationUnit(dna=self.dna)
            new_unit.energy = self.energy # 继承能量
            spawn_unit(board, x, y, new_unit)
            return

        if self.energy <= 0.1: # 能量太少不传输
            return

        neighbors = get_neighbors(board, x, y)
        if not neighbors:
            return

        # 寻找周围有生物的格子
        targets = []
        for nx, ny in neighbors:
            target_org = board[nx][ny].organism
            if target_org:
                targets.append(target_org)
        
        if targets:
            # 传输自身大部分能量，例如 80%
            transfer_total = self.energy * 0.8
            self.energy -= transfer_total
            
            # 平均分配给所有邻居
            share = transfer_total / len(targets)
            received_amount = share * 1
            
            for org in targets:
                org.energy += received_amount

class CombatUnit(OrganismUnit):
    """战斗单元: 攻击异种生物并同化"""
    def __init__(self, dna=None):
        super().__init__(dna)
        self.patience = 5 # 战斗状态维持回合数 (滞留时间)
        self.alert_target = None # 警报目标 (x, y)

    def __repr__(self):
        return f"CombatUnit(E={self.energy:.1f})"

    def respond_to_alert(self, board, x, y, enemy_pos):
        # 收到警报，设定目标并重置耐心
        self.alert_target = enemy_pos
        self.patience = 10

    def update(self, board, x, y):
        # 基础代谢 (战斗单元消耗更高，维持军队需要代价)
        self.energy -= 1.0
        if self.energy <= 0:
            # 死亡由主循环处理，但这里可以提前返回
            return

        neighbors = get_neighbors(board, x, y)
        my_species = self.dna.get('species_id', 0)
        has_enemy = False
        
        # 1. 优先攻击周围的敌人
        for nx, ny in neighbors:
            target_unit = board[nx][ny].organism
            if target_unit:
                target_species = target_unit.dna.get('species_id', 0)
                if target_species != my_species:
                    has_enemy = True
                    
                    # 发出警报 (呼叫支援)
                    alert_allies(board, x, y, my_species, (nx, ny))
                    # 敌人也发出警报
                    alert_allies(board, nx, ny, target_species, (x, y))
                    
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
                            spawn_unit(board, nx, ny, new_unit)
                        else:
                            # 仅杀死，不占领
                            kill_unit(board, nx, ny)
                    
                    if self.energy <= 0:
                        kill_unit(board, x, y)
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
                    if (x, y) == self.alert_target:
                        self.alert_target = None
                    else:
                        # 简单的贪婪寻路向警报点移动
                        empty_neighbors = [n for n in neighbors if board[n[0]][n[1]].organism is None]
                        if empty_neighbors:
                            best_spot = min(empty_neighbors, key=lambda p: (p[0]-self.alert_target[0])**2 + (p[1]-self.alert_target[1])**2)
                            move_to = best_spot
                            
                            # 如果距离很近了，可能已经到达附近，清除目标以防卡死
                            if (x-self.alert_target[0])**2 + (y-self.alert_target[1])**2 < 2:
                                self.alert_target = None

                # 2. 猎杀模式 (如果没有警报目标)
                if not move_to:
                    # 猎杀模式: 扫描半径2的范围寻找敌人
                    height = len(board)
                    width = len(board[0])
                    target_pos = None
                    
                    # 简单的范围2扫描
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if abs(dx) <= 1 and abs(dy) <= 1: continue # 跳过邻居(已检查)
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < height and 0 <= ny < width:
                                u = board[nx][ny].organism
                                if u and u.dna.get('species_id', 0) != my_species:
                                    target_pos = (nx, ny)
                                    break
                        if target_pos: break
                    
                    empty_neighbors = [n for n in neighbors if board[n[0]][n[1]].organism is None]
                    
                    if target_pos and empty_neighbors:
                        # 向敌人方向移动
                        best_dist = 999
                        for ex, ey in empty_neighbors:
                            dist = ((ex - target_pos[0])**2 + (ey - target_pos[1])**2)
                            if dist < best_dist:
                                best_dist = dist
                                move_to = (ex, ey)
                    elif empty_neighbors:
                        # 随机巡逻
                        move_to = random.choice(empty_neighbors)
                
                if move_to:
                    nx, ny = move_to
                    me = self
                    kill_unit(board, x, y) 
                    spawn_unit(board, nx, ny, me)
                    return

        # 如果倒计时结束且无敌人，变回复制单元以继续发展
        if self.patience <= 0:
            new_unit = ReplicationUnit(dna=self.dna)
            new_unit.energy = self.energy
            spawn_unit(board, x, y, new_unit)

class FactoryUnit(OrganismUnit):
    """工厂单元: 消耗能量增加周围肥力"""
    def __repr__(self):
        return f"FactoryUnit(E={self.energy:.1f})"

    def update(self, board, x, y):
        # 消耗能量
        consumption = 1.0
        if self.energy < consumption:
            # 能量不足，无法工作，甚至可能死亡(由主循环处理)
            return
            
        self.energy -= consumption
        
        # 增加周围3格(半径3)的肥力
        height = len(board)
        width = len(board[0])
        
        # 转换效率: 1点能量 -> 0.01 肥力 (分摊给周围)
        # 或者每个格子增加固定值
        # 恢复缓慢: 0.01
        fertility_gain = 0.01
        
        for i in range(-3, 4):
            for j in range(-3, 4):
                if i == 0 and j == 0: continue
                nx, ny = x + i, y + j
                if 0 <= nx < height and 0 <= ny < width:
                    board[nx][ny].fertility += fertility_gain

# 定义棋盘格子类
class Cell:
    def __init__(self):
        # 光照（浮点数，范围0~1）
        self.light = random.random()
        # 土地肥力（浮点数，范围0~1）
        self.base_fertility = random.random()
        self.fertility = self.base_fertility
        # 其上生物（默认为无）
        self.organism = None
        # 是否为虚空 (不可通行区域)
        self.is_void = False

    def __repr__(self):
        org_str = self.organism if self.organism else "None"
        return f"Cell(Light={self.light:.2f}, Fertility={self.fertility:.2f}, Organism={org_str})"

def spawn_unit(board, x, y, unit, apply_fertility=True):
    if board[x][y].is_void:
        return # 无法在虚空中生成
        
    board[x][y].organism = unit
    height = len(board)
    width = len(board[0])
    
    # 肥力惩罚: 周围8格 -0.05
    if apply_fertility:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                nx, ny = x + i, y + j
                if 0 <= nx < height and 0 <= ny < width:
                    board[nx][ny].fertility -= 0.05
                
    # 光照惩罚: 光合单元周围3格减半
    if isinstance(unit, PhotosynthesisUnit):
        for i in range(-3, 4):
            for j in range(-3, 4):
                if i == 0 and j == 0: continue
                nx, ny = x + i, y + j
                if 0 <= nx < height and 0 <= ny < width:
                    board[nx][ny].light *= 0.5

def kill_unit(board, x, y):
    """移除生物并恢复环境影响"""
    unit = board[x][y].organism
    if unit is None:
        return
    
    height = len(board)
    width = len(board[0])
    
    # 肥力恢复: 周围8格 +0.05
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            nx, ny = x + i, y + j
            if 0 <= nx < height and 0 <= ny < width:
                board[nx][ny].fertility += 0.05

    # 光照恢复: 光合单元周围3格恢复
    if isinstance(unit, PhotosynthesisUnit):
        for i in range(-3, 4):
            for j in range(-3, 4):
                if i == 0 and j == 0: continue
                nx, ny = x + i, y + j
                if 0 <= nx < height and 0 <= ny < width:
                    board[nx][ny].light *= 2.0
    
    board[x][y].organism = None

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + n10*t[:,:,0]
    n1 = n01*(1-t[:,:,0]) + n11*t[:,:,0]
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    max_value = 0
    
    for _ in range(octaves):
        # Calculate current resolution for this octave
        current_res = (int(res[0] * frequency), int(res[1] * frequency))
        
        # Ensure resolution is at least 1
        current_res = (max(1, current_res[0]), max(1, current_res[1]))
        
        # Pad shape to be divisible by current_res
        pad_x = (current_res[0] - (shape[0] % current_res[0])) % current_res[0]
        pad_y = (current_res[1] - (shape[1] % current_res[1])) % current_res[1]
        padded_shape = (shape[0] + pad_x, shape[1] + pad_y)
        
        # Generate noise
        octave_noise = generate_perlin_noise_2d(padded_shape, current_res)
        
        # Crop back to original shape
        octave_noise = octave_noise[:shape[0], :shape[1]]
        
        noise += octave_noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
        
    return noise / max_value

def generate_board(width, height, mode='radial'):
    """生成二维棋盘"""
    print(f"正在生成 {width}x{height} 的棋盘 (模式: {mode})...")
    board = [[Cell() for _ in range(width)] for _ in range(height)]
    
    if mode == 'sparse':
        # 1. 初始化全为虚空
        for r in range(height):
            for c in range(width):
                board[r][c].is_void = True
                board[r][c].light = 0.0
                board[r][c].base_fertility = 0.0
                board[r][c].fertility = 0.0
        
        # 2. 生成随机节点
        num_nodes = int((width * height) ** 0.5) * 2
        nodes = []
        for _ in range(num_nodes):
            nodes.append((random.randint(5, height-6), random.randint(5, width-6)))
            
        # 3. 连接节点 (MST + K近邻，确保单连通且有环)
        edges = set()
        
        # 3.1 准备边列表用于MST
        all_possible_edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                r1, c1 = nodes[i]
                r2, c2 = nodes[j]
                d = (r1-r2)**2 + (c1-c2)**2
                all_possible_edges.append((d, i, j))
        
        all_possible_edges.sort()
        
        # 3.2 Kruskal算法生成最小生成树 (保证全局连通)
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
            
        for _, u, v in all_possible_edges:
            if union(u, v):
                edges.add(tuple(sorted((u, v))))
                
        # 3.3 叠加K近邻连接 (增加局部环路，避免单纯的树状结构)
        for i in range(len(nodes)):
            distances = []
            for j in range(len(nodes)):
                if i == j: continue
                r1, c1 = nodes[i]
                r2, c2 = nodes[j]
                d = (r1-r2)**2 + (c1-c2)**2
                distances.append((d, j))
            
            distances.sort()
            # 连接最近的2个点
            for k in range(min(2, len(distances))):
                j = distances[k][1]
                edges.add(tuple(sorted((i, j))))
        
        # 4. 栅格化路径
        for i, j in edges:
            r1, c1 = nodes[i]
            r2, c2 = nodes[j]
            
            dist = int(max(abs(r2-r1), abs(c2-c1)))
            if dist == 0: continue
            
            for step in range(dist + 1):
                t = step / dist
                r = int(r1 + (r2 - r1) * t)
                c = int(c1 + (c2 - c1) * t)
                
                # 绘制路径 (宽度为3)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            cell = board[nr][nc]
                            cell.is_void = False
                            # 路径上资源丰富
                            cell.light = random.uniform(0.6, 1.0)
                            cell.base_fertility = random.uniform(0.6, 1.0)
                            cell.fertility = cell.base_fertility

    if mode == 'perlin':
        try:
            # 使用分形噪声生成更自然的网状结构
            # 基础分辨率 (低频)
            base_res = (4, 4)
            # 4个八度叠加，增加细节
            noise = generate_fractal_noise_2d((height, width), base_res, octaves=4, persistence=0.5)
            
            # 归一化到 0-1
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # 制造网状效果 (Ridge Noise): 1 - abs(2*noise - 1)
            # 这会产生类似血管/网状的结构
            net_noise = 1.0 - np.abs(2.0 * noise - 1.0)
            
            # 增加对比度，让网格更明显，背景更黑
            net_noise = net_noise ** 2.0
            
            # 再次归一化
            net_noise = (net_noise - net_noise.min()) / (net_noise.max() - net_noise.min())
            
            for r in range(height):
                for c in range(width):
                    val = net_noise[r][c]
                    board[r][c].light = val
                    board[r][c].base_fertility = val
                    board[r][c].fertility = val
        except Exception as e:
            print(f"柏林噪声生成失败，回退到径向模式: {e}")
            mode = 'radial'

    if mode == 'radial':
        # 设置光照为中心向外递减
        center_r, center_c = height / 2.0, width / 2.0
        max_dist = ((height / 2.0)**2 + (width / 2.0)**2)**0.5
        
        for r in range(height):
            for c in range(width):
                dist = ((r - center_r)**2 + (c - center_c)**2)**0.5
                # 线性递减: 中心1.0 -> 角落0.0
                val = max(0.0, 1.0 - (dist / max_dist))
                board[r][c].light = val
                # 初始肥力与光照相同
                board[r][c].base_fertility = val
                board[r][c].fertility = val

    print("棋盘生成完毕。")
    return board

def save_top_species(board, filename="top_species.json"):
    print("正在保存前10名物种DNA...")
    species_counts = {}
    species_dnas = {}
    
    height = len(board)
    width = len(board[0])
    
    for r in range(height):
        for c in range(width):
            unit = board[r][c].organism
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

def populate_randomly(board, count=100, initial_dnas=None):
    """随机放置一些生物单元用于演示"""
    height = len(board)
    width = len(board[0])
    
    # 辅助函数: 寻找有效位置
    def get_valid_spot():
        for _ in range(50): # 尝试50次
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            if not board[r][c].is_void and board[r][c].organism is None:
                return r, c
        return None, None

    if initial_dnas:
        print(f"使用存档的 {len(initial_dnas)} 个DNA初始化...")
        for dna in initial_dnas:
             r, c = get_valid_spot()
             if r is not None:
                 spawn_unit(board, r, c, SporeUnit(dna=dna))
        
        # 如果不足10个，补齐随机
        remaining = 10 - len(initial_dnas)
        if remaining > 0:
             for _ in range(remaining):
                r, c = get_valid_spot()
                if r is not None:
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
                    spawn_unit(board, r, c, SporeUnit(dna=dna))
    else:
        # 放置一些孢子单元
        for _ in range(10):
            r, c = get_valid_spot()
            if r is not None:
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
                    'hub_behavior': random.uniform(0.0, 1.0),
                    'corridor_behavior': random.uniform(0.0, 1.0),
                    'species_id': random.randint(1, 100)
                }
                spawn_unit(board, r, c, SporeUnit(dna=dna))

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
            elif 'threshold' in key:
                if 'replication' in key: # 能量阈值
                    new_value = max(10.0, new_value)
                else: # 其他阈值 0-1
                    new_value = max(0.0, min(1.0, new_value))
            
            new_dna[key] = new_value
    return new_dna

def perform_pollination(board):
    height = len(board)
    width = len(board[0])
    
    # 1. 随机选取源位置
    # 尝试寻找有效生物
    source_unit = None
    for _ in range(20):
        r_src = random.randint(0, height - 1)
        c_src = random.randint(0, width - 1)
        if board[r_src][c_src].organism:
            source_unit = board[r_src][c_src].organism
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
            'hub_behavior': random.uniform(0.0, 1.0),
            'corridor_behavior': random.uniform(0.0, 1.0),
            'species_id': random.randint(1, 100)
        }
        
    # 2. 寻找目标空位 (尝试几次)
    for _ in range(20):
        r_dst = random.randint(0, height - 1)
        c_dst = random.randint(0, width - 1)
        # 必须是非虚空且为空
        if not board[r_dst][c_dst].is_void and board[r_dst][c_dst].organism is None:
            # 3. 生成变异孢子
            new_dna = mutate_dna(base_dna)
            spawn_unit(board, r_dst, c_dst, SporeUnit(dna=new_dna))
            break

def simulate_step(board, step_count=0):
    """执行一步模拟"""
    height = len(board)
    width = len(board[0])
    
    # 传粉机制 (每10步)
    if step_count % 10 == 0:
        perform_pollination(board)
    
    # 1. 肥力恢复与致死检查
    for r in range(height):
        for c in range(width):
            cell = board[r][c]
            
            # 肥力恢复: 缓慢回升到 base_fertility
            if cell.fertility < cell.base_fertility:
                cell.fertility += 0.005 # 恢复速度
                if cell.fertility > cell.base_fertility:
                    cell.fertility = cell.base_fertility
            
            # 致死检查: 肥力 < 0 杀死生物
            if cell.fertility < 0 and cell.organism:
                kill_unit(board, r, c)

    # 2. 遍历所有格子进行更新
    # 注意：为了简单起见，这里是顺序更新。更严谨的模拟可能需要随机顺序或双缓冲。
    for r in range(height):
        for c in range(width):
            unit = board[r][c].organism
            if unit:
                # 能量检查: 能量 < 0 死亡
                if unit.energy < 0:
                    kill_unit(board, r, c)
                    continue
                
                unit.update(board, r, c)

def get_board_rgb(board):
    """将棋盘转换为RGB矩阵用于可视化"""
    height = len(board)
    width = len(board[0])
    # 初始化为黑色背景
    grid = np.zeros((height, width, 3))
    
    for r in range(height):
        for c in range(width):
            cell = board[r][c]
            if cell.organism:
                if isinstance(cell.organism, PhotosynthesisUnit):
                    grid[r, c] = [0, 1, 0] # 绿色: 光合单元
                elif isinstance(cell.organism, ReplicationUnit):
                    grid[r, c] = [1, 0, 0] # 红色: 复制单元
                elif isinstance(cell.organism, ConnectionUnit):
                    grid[r, c] = [0, 0, 1] # 蓝色: 连接单元
                elif isinstance(cell.organism, FactoryUnit):
                    grid[r, c] = [1, 0, 1] # 紫色: 工厂单元
                elif isinstance(cell.organism, CombatUnit):
                    grid[r, c] = [1, 0.5, 0] # 橙色: 战斗单元
            else:
                # 背景显示
                if hasattr(cell, 'is_void') and not cell.is_void:
                    grid[r, c] = [0.15, 0.15, 0.15] # 有效路径显示为深灰色
                else:
                    grid[r, c] = [0, 0, 0] # 虚空显示为黑色
    return grid

def get_dna_rgb(board):
    """将棋盘转换为RGB矩阵用于可视化DNA差异"""
    height = len(board)
    width = len(board[0])
    grid = np.zeros((height, width, 3))
    
    for r in range(height):
        for c in range(width):
            cell = board[r][c]
            if cell.organism:
                dna = cell.organism.dna
                # light_threshold: 0.2 ~ 0.8 -> Red channel
                r_val = dna.get('light_threshold', 0.0)
                
                # replication_threshold: 20 ~ 80 -> Green channel
                rep_thresh = dna.get('replication_threshold', 50.0)
                # Normalize 20-80 to 0-1
                g_val = (rep_thresh - 20.0) / 60.0 
                g_val = max(0.0, min(1.0, g_val))
                
                # fertility_threshold: 0.2 ~ 0.8 -> Blue channel
                fert_thresh = dna.get('fertility_threshold', 0.5)
                b_val = fert_thresh
                
                grid[r, c] = [r_val, g_val, b_val]
            else:
                # 背景显示
                if hasattr(cell, 'is_void') and not cell.is_void:
                    grid[r, c] = [0.15, 0.15, 0.15] # 有效路径显示为深灰色
                else:
                    grid[r, c] = [0, 0, 0] # 虚空显示为黑色
    return grid

def get_light_rgb(board):
    """将棋盘光照转换为RGB矩阵"""
    height = len(board)
    width = len(board[0])
    grid = np.zeros((height, width, 3))
    for r in range(height):
        for c in range(width):
            val = board[r][c].light
            grid[r, c] = [val, val, val] # 灰度显示光照
    return grid

def get_fertility_rgb(board):
    """将棋盘肥力转换为RGB矩阵"""
    height = len(board)
    width = len(board[0])
    grid = np.zeros((height, width, 3))
    for r in range(height):
        for c in range(width):
            val = board[r][c].fertility
            # 肥力可能为负，进行可视化处理
            # 正值: 绿色 (0~1)
            # 负值: 红色 (-1~0)
            if val >= 0:
                grid[r, c] = [0, min(1.0, val), 0]
            else:
                grid[r, c] = [min(1.0, abs(val)), 0, 0]
    return grid

def get_energy_rgb(board):
    """将棋盘能量转换为RGB矩阵"""
    height = len(board)
    width = len(board[0])
    grid = np.zeros((height, width, 3))
    for r in range(height):
        for c in range(width):
            unit = board[r][c].organism
            if unit:
                # 能量可视化: 黄色 (R+G)
                # 假设 100 为显示上限
                val = min(1.0, max(0.0, unit.energy / 100.0))
                grid[r, c] = [val, val, 0]
            else:
                grid[r, c] = [0, 0, 0]
    return grid

def run_batch_simulation(rounds=10, steps=1000, width=100, height=100):
    """运行无可视化的批量模拟"""
    print(f"启动批量模拟模式: 共 {rounds} 轮, 每轮 {steps} 帧")
    
    for r in range(1, rounds + 1):
        print(f"\n--- 第 {r} / {rounds} 轮 ---")
        # 1. 生成棋盘
        board = generate_board(width, height)
        
        # 2. 读取存档 (上一轮的优胜者)
        saved_dnas = load_top_species()
        
        # 3. 初始生物
        populate_randomly(board, count=200, initial_dnas=saved_dnas)
        
        # 4. 模拟循环
        print("正在模拟...")
        for s in range(steps):
            simulate_step(board, step_count=s)
            if (s + 1) % 100 == 0:
                print(f"  进度: {s + 1} / {steps}")
        
        # 5. 保存结果
        save_top_species(board)
        print(f"第 {r} 轮完成，已保存优势物种存档。")
        
    print("\n所有模拟轮次结束。")

if __name__ == "__main__":
    # 检查命令行参数是否为 batch
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        run_batch_simulation()
        sys.exit()

    # 提示: 1000x1000 的纯 Python 模拟会非常慢，建议使用较小尺寸进行可视化演示
    # 或者需要将核心逻辑重写为 numpy 矩阵运算
    WIDTH = 100 
    HEIGHT = 100
    
    # 1. 生成棋盘
    # 检查命令行参数是否指定了生成模式
    gen_mode = 'sparse' # 默认为稀疏图模式
    if len(sys.argv) > 1:
        if 'perlin' in sys.argv:
            gen_mode = 'perlin'
        elif 'radial' in sys.argv:
            gen_mode = 'radial'
        
    game_board = generate_board(WIDTH, HEIGHT, mode=gen_mode)
    
    # 读取存档
    saved_dnas = load_top_species()
    
    # 2. 随机放置一些生物进行演示
    populate_randomly(game_board, count=200, initial_dnas=saved_dnas)
    
    print("初始化可视化窗口...")
    
    # 设置图形: 2x3 子图
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax5 = axs[0]
    ax3, ax4, ax6 = axs[1]
    
    ax1.set_title("Organism Types")
    ax2.set_title("DNA (R:Light, G:Rep, B:Fert)")
    ax5.set_title("Energy Levels (Yellow)")
    
    ax3.set_title("Light Distribution")
    ax4.set_title("Fertility (Green>0, Red<0)")
    ax6.set_title("Top 5 Species")
    ax6.axis('off')
    
    # 初始化排行榜文本对象
    ranking_texts = []
    for i in range(5):
        # 使用相对坐标 (0~1)
        txt = ax6.text(0.1, 0.8 - i * 0.15, "", fontsize=12, fontweight='bold')
        ranking_texts.append(txt)

    # 初始图像
    im1 = ax1.imshow(get_board_rgb(game_board), interpolation='nearest')
    im2 = ax2.imshow(get_dna_rgb(game_board), interpolation='nearest')
    im5 = ax5.imshow(get_energy_rgb(game_board), interpolation='nearest')
    
    im3 = ax3.imshow(get_light_rgb(game_board), interpolation='nearest', vmin=0, vmax=1)
    im4 = ax4.imshow(get_fertility_rgb(game_board), interpolation='nearest')
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')
    
    def animate(frame):
        # 执行一步模拟
        simulate_step(game_board, step_count=frame)
        
        # 更新图像
        im1.set_array(get_board_rgb(game_board))
        im2.set_array(get_dna_rgb(game_board))
        im5.set_array(get_energy_rgb(game_board))
        im3.set_array(get_light_rgb(game_board))
        im4.set_array(get_fertility_rgb(game_board))
        
        # 统计种群数量
        species_counts = {}
        species_sample_dna = {}
        
        height = len(game_board)
        width = len(game_board[0])
        for r in range(height):
            for c in range(width):
                unit = game_board[r][c].organism
                if unit:
                    s_id = unit.dna.get('species_id', 0)
                    species_counts[s_id] = species_counts.get(s_id, 0) + 1
                    if s_id not in species_sample_dna:
                        species_sample_dna[s_id] = unit.dna
        
        # 排序并取前5
        sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 更新排行榜文本
        for i in range(5):
            txt = ranking_texts[i]
            if i < len(sorted_species):
                s_id, count = sorted_species[i]
                name = get_species_name(s_id)
                dna = species_sample_dna[s_id]
                
                # 计算代表色 (与DNA图层一致)
                r_val = dna.get('light_threshold', 0.0)
                rep_thresh = dna.get('replication_threshold', 50.0)
                g_val = max(0.0, min(1.0, (rep_thresh - 20.0) / 60.0))
                b_val = dna.get('fertility_threshold', 0.5)
                color = (r_val, g_val, b_val)
                
                txt.set_text(f"{name}: {count}")
                txt.set_color(color)
            else:
                txt.set_text("")
        
        fig.suptitle(f"Ecosystem Simulation - Round {frame}")
        return [im1, im2, im3, im4, im5] + ranking_texts

    # 创建动画
    ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
    
    # 交互控制
    is_paused = False
    
    def on_key(event):
        global is_paused
        if event.key == ' ':
            if is_paused:
                ani.event_source.start()
                is_paused = False
            else:
                ani.event_source.stop()
                is_paused = True
                fig.suptitle("Ecosystem Simulation - Paused")
                fig.canvas.draw_idle()
        elif event.key == 'q' or event.key == 'escape':
            save_top_species(game_board)
            plt.close(fig)
            print("模拟结束")

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("开始模拟。请查看弹出的窗口。")
    print("控制说明: 按 [空格] 暂停/继续，按 [q] 或 [Esc] 退出。")
    plt.show()

