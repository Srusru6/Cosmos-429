import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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
            'fertility_sacrifice_threshold': 0.1 # 肥力低于此值时，连接单元牺牲自己
        }

    def update(self, board, x, y):
        # 基础代谢消耗
        self.energy -= 0.5
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
                neighbors.append((nx, ny))
    return neighbors

# 定义具体生物单元类
class PhotosynthesisUnit(OrganismUnit):
    """光合单元: 将光照转换为周围单元的能量"""
    def __repr__(self):
        return f"PhotosynthesisUnit(E={self.energy:.1f})"

    def update(self, board, x, y):
        # 获取当前格子的光照
        light = board[x][y].light
        
        # 线性公式转换能量 (例如系数为 100)
        generated_energy = light * 100.0
        
        neighbors = get_neighbors(board, x, y)
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

    def __repr__(self):
        return f"ReplicationUnit(E={self.energy:.1f}, Acc={self.accumulated_energy:.1f})"

    def update(self, board, x, y):
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
            self.trigger_event(board, x, y)
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
        if empty_neighbors:
            nx, ny = random.choice(empty_neighbors)
            # 传递DNA给后代
            spawn_unit(board, nx, ny, ReplicationUnit(dna=self.dna))
            # print(f"复制单元在 ({x}, {y}) 繁殖到了 ({nx}, {ny})")

        # 2. 将自己变为 光合单元 或 连接单元
        # 根据DNA中的光照阈值和肥力阈值决定
        current_light = board[x][y].light
        current_fertility = board[x][y].fertility
        
        light_thresh = self.dna.get('light_threshold', 0.5)
        fert_thresh = self.dna.get('fertility_threshold', 0.5)
        
        # 只有当光照和肥力都达标时，才成为光合单元
        if current_light > light_thresh and current_fertility > fert_thresh:
            new_type = PhotosynthesisUnit
        else:
            new_type = ConnectionUnit
            
        spawn_unit(board, x, y, new_type(dna=self.dna))
        # print(f"复制单元 ({x}, {y}) 变身为 {new_type.__name__}")

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
    """连接单元: 将能量按0.9递减传递给周围单元"""
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

    def __repr__(self):
        org_str = self.organism if self.organism else "None"
        return f"Cell(Light={self.light:.2f}, Fertility={self.fertility:.2f}, Organism={org_str})"

def spawn_unit(board, x, y, unit, apply_fertility=True):
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

def generate_board(width, height):
    """生成二维棋盘"""
    print(f"正在生成 {width}x{height} 的棋盘...")
    board = [[Cell() for _ in range(width)] for _ in range(height)]
    
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

def populate_randomly(board, count=100):
    """随机放置一些生物单元用于演示"""
    height = len(board)
    width = len(board[0])
    
    # 放置一些孢子单元
    for _ in range(10):
        r = random.randint(10, height - 11)
        c = random.randint(10, width - 11)
        # 随机DNA阈值
        dna = {
            'light_threshold': random.uniform(0.2, 0.8),
            'fertility_threshold': random.uniform(0.2, 0.8),
            'replication_threshold': random.uniform(20.0, 80.0),
            'mutation_prob': random.uniform(0.0, 0.05), # 0% ~ 5%
            'degeneration_prob': random.uniform(0.0, 0.05), # 0% ~ 5%
            'fertility_sacrifice_threshold': random.uniform(0.0, 0.3)
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
    r_src = random.randint(0, height - 1)
    c_src = random.randint(0, width - 1)
    source_unit = board[r_src][c_src].organism
    
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
            'fertility_sacrifice_threshold': random.uniform(0.0, 0.3)
        }
        
    # 2. 寻找目标空位 (尝试几次)
    for _ in range(10):
        r_dst = random.randint(0, height - 1)
        c_dst = random.randint(0, width - 1)
        if board[r_dst][c_dst].organism is None:
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
            else:
                # 背景显示黑色
                grid[r, c] = [0, 0, 0]
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
                # 背景显示黑色
                grid[r, c] = [0, 0, 0]
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

if __name__ == "__main__":
    # 提示: 1000x1000 的纯 Python 模拟会非常慢，建议使用较小尺寸进行可视化演示
    # 或者需要将核心逻辑重写为 numpy 矩阵运算
    WIDTH = 100 
    HEIGHT = 100
    
    # 1. 生成棋盘
    game_board = generate_board(WIDTH, HEIGHT)
    
    # 2. 随机放置一些生物进行演示
    populate_randomly(game_board, count=200)
    
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
    ax6.axis('off') # 第6个暂时不用
    
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
        
        fig.suptitle(f"Ecosystem Simulation - Round {frame}")
        return [im1, im2, im3, im4, im5]

    # 创建动画
    ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
    
    print("开始模拟。请查看弹出的窗口。")
    plt.show()

