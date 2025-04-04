# src/models.py
import numpy as np
from scipy.optimize import minimize

class Provider:
    def __init__(self, f_max, B_max, c_E, c_N):
        self.f_max = f_max  # 总计算能力
        self.B_max = B_max  # 总带宽
        self.c_E = c_E     # ESP 成本
        self.c_N = c_N     # NSP 成本

    def print_providers(self):
      print("f_{max}:",self.f_max)
      print("B_{max}:",self.B_max)
      print("c_E:",self.c_E)
      print("c_N:",self.c_N)

class Task:
    def __init__(self, d, b, alpha):
        """
        d: CPU周期数 (计算工作量)
        b: 数据量 (Bytes)
        alpha: 延时敏感度
        """
        self.d = d
        self.b = b
        self.alpha = alpha

class Channel:
    def __init__(self, trans_power, channel_gain, background_noise) -> None:
        self.trans_power = trans_power
        self.channel_gain = channel_gain
        self.background_noise = background_noise
        self.SNR = np.log2(1+trans_power*channel_gain/background_noise)

class User:
    def __init__(self, user_id, task, local_cpu, channel):
        self.user_id = user_id
        self.task = task
        self.channel = channel
        self.local_cpu = local_cpu  # 本地计算能力
        self.S_i = channel.SNR # 传输系数

    def local_computation_time(self):
        # 公式 (1): T_li = d_i / f_li
        return self.task.d / self.local_cpu

    def offloading_time(self,f_i,B_i):
        # 公式 (2) & (3)：T_ui 和 T_ci 的组合，此处假设 f_i 和 B_i 已知
        # 此处仅给出示意
        if f_i <= 1e-9 or B_i <= 1e-9:
          return float('inf')  # 返回一个很大的值
        T_u = self.task.b / (B_i*self.S_i)  if B_i > 0 else np.inf
        T_c = self.task.d / f_i  if f_i > 0 else np.inf
        return T_u + T_c

    def cost_local(self):
        return self.task.alpha * self.local_computation_time()

    def cost_offload(self, p_E, p_N, f_i, B_i):
        # 公式 (6): C_ei = alpha_i * T_ei + p_E * f_i + p_N * B_i
        T_e = self.offloading_time(f_i,B_i)
        return self.task.alpha * T_e + p_E * f_i + p_N * B_i

    def C_hat_e_i(self, p_E, p_N):
      C_hat_eb_i = 2 * np.sqrt(self.task.alpha * self.task.b * p_N / self.S_i)
      C_hat_ef_i = 2 * np.sqrt(self.task.alpha * self.task.d * p_E)
      return C_hat_ef_i+C_hat_eb_i

    def C_hat_eb_i(self, p_N):
      return 2 * np.sqrt(self.task.alpha * self.task.b * p_N / self.S_i)

    def C_hat_ef_i(self, p_E):
      return 2 * np.sqrt(self.task.alpha * self.task.d * p_E)

    def f_hat(self, p_E):
      return np.sqrt(self.task.alpha * self.task.d / p_E)

    def B_hat(self, p_N):
      return np.sqrt(self.task.alpha * self.task.b / (self.S_i * p_N))

    def f_thres(self, p_E, p_N):
      C_l_i = self.task.alpha * (self.task.d / self.local_cpu)
      C_hat_eb_i = 2 * np.sqrt(self.task.alpha * self.task.b * p_N / self.S_i)
      C_hat_ef_i = 2 * np.sqrt(self.task.alpha * self.task.d * p_E)
      f_thresh = ((C_l_i - C_hat_eb_i) - np.sqrt((C_l_i - C_hat_eb_i)**2 - C_hat_ef_i**2)) / (2*p_E)
      return f_thresh
      
    def B_thres(self, p_E, p_N):
      C_l_i = self.task.alpha * (self.task.d / self.local_cpu)
      C_hat_eb_i = 2 * np.sqrt(self.task.alpha * self.task.b * p_N / self.S_i)
      C_hat_ef_i = 2 * np.sqrt(self.task.alpha * self.task.d * p_E)
      B_thresh = ((C_l_i - C_hat_ef_i) - np.sqrt((C_l_i - C_hat_ef_i)**2 - C_hat_eb_i**2)) / (2*p_N)
      return B_thresh

    def print_bounds(self, p_E, p_N):
      # print("user id:", self.user_id)
      print(" f_{thres}_", self.user_id, "=", self.f_thres(p_E,p_N), " B_{thres}_", self.user_id, "=", self.B_thres(p_E,p_N))
      print(" f_{hat}_", self.user_id, "=", self.f_hat(p_E), " B_{hat}_", self.user_id, "=", self.B_hat(p_N))

    def print_user(self):
      # print("user id:", self.user_id)
      print(" task: d=", self.task.d, " b=", self.task.b," alpha=", self.task.alpha)
      print(" local cpu=", self.local_cpu)
      print(" S_i=", self.S_i)
      # print("channel: trans_power=", self.channel.trans_power, " channel_gain=", self.channel.channel_gain, " noise=", self.channel.background_noise)

    def print_costs(self, p_E, p_N):
      # print("user id:", self.user_id)
      print(" C^l_", self.user_id, "=", self.cost_local(), " C_hat_e_", self.user_id, "=", self.C_hat_e_i(p_E,p_N))
      print(" C_{hat}_ef_", self.user_id, "=", self.C_hat_ef_i(p_E), " C_{hat}_eb_", self.user_id, "=", self.C_hat_eb_i(p_N))

    def log_info(self, p_E, p_N):
      info = {
        "user id": self.user_id,
        "task": [self.task.d,self.task.b,self.task.alpha],
        "S_i": self.S_i,
        "local_cpu": self.local_cpu,
        "C_l_i": self.cost_local(),
        "C_hat_ei": self.C_hat_e_i(p_E, p_N),
        "C_hat_ef_i": self.C_hat_ef_i(p_E),
        "C_hat_eb_i": self.C_hat_eb_i(p_N),
        "f_thres": self.f_thres(p_E, p_N),
        "B_thres": self.B_thres(p_E,p_N),
        "f_hat": self.f_hat(p_E),
        "B_hat": self.B_hat(p_N)
      }
      return info

    def offload_best_response(self, p_E, p_N, remain_f, remain_B):
      tilde_f = min(remain_f, self.f_hat(p_E))
      tilde_B = min(remain_B, self.B_hat(p_N))
      if self.cost_offload(p_E, p_N, tilde_f, tilde_B) <= self.cost_local():
        return 1
      else:
        return 0

    def is_allocation_feasible(self,f_i,B_i,p_E,p_N):
      if f_i < self.f_thres or f_i > self.f_hat:
        return False
      if B_i < self.B_thres or B_i > self.B_hat:
        return False
      if self.cost_offload(p_E, p_N, f_i, B_i) > self.cost_local():
        return False
      return True


# 其他模型函数：例如成本函数的梯度、最优点求解公式等，可根据论文推导添加

def is_offloader_set_feasible(users, provider, X, p_E, p_N, verbose=False):
  """
  判断X在给定价格p_E, p_N下是否有可行的资源分配
  需要确保输入的p_E, p_N本身属于P_X
  """
  # 如果 X 是空集，直接返回not feasible
  if not X : 
    if verbose: print("is_offloader_set_feasible: X is empty set.")
    return True
  # 提取 offloader 用户列表（即 user.user_id in X）
  users_X = [user for user in users if user.user_id in X]

  # 调用 ORA 求解器，求解 offloader 用户在当前价格下的最优资源分配
  opt_f, opt_B, opt_cost = ora_solver(users_X, provider, p_E, p_N)
  if opt_f is None or opt_cost is None:
      # ORA 无法求解或不可行
      if verbose: print("is_offloader_set_feasible: unable to solve ORA(X)")
      return False
  return True

def is_allocation_feasible(users, provider, f, B, p_E, p_N):
  if np.sum(f) > provider.f_max: return False
  if np.sum(B) > provider.B_max: return False
  for i,user in enumerate(users):
    if user.is_allocation_feasible(f[i],B[i],p_E,p_N) != True:
      return False
  return True

def social_cost(users, X, f, B, p_E, p_N):
    """
    计算社会成本：
    - 对于集合 X 中的用户，优化 (f_i, B_i) 使得 C_i^e 最小
    - 对于未选择卸载的用户，计算本地执行成本 C_j^l
    """
    total_cost = 0

    for i,user in enumerate(users):
        if i in X:  # 选择卸载
            # 计算卸载成本
            cost = user.cost_offload(p_E, p_N, f[i], B[i])
        else:  # 本地执行
            cost = user.cost_local()

        total_cost += cost

    return total_cost

def is_price_feasible(users, provider, X, p_E, p_N, verbose=False):
    """
    检查给定价格 (p_E, p_N) 是否属于价格可行域 P_X，
    条件包括：
      1. 对 offloader 用户集合 X，ORA(X) 可行，即存在一组资源分配 {f_i, B_i} 使得
         对于每个 i∈X，卸载成本 C_e_i(f_i,B_i) <= 本地计算成本 C_l_i。
      2. 对于不在 X 中的用户 j，基于闭式估计的 offloading 成本必须高于本地成本加上一个正的 epsilon，
         即 C_e_j >= C_l_j + epsilon。
         
    参数：
      p_E, p_N: 当前候选的 ESP 和 NSP 定价
      users: 完整的用户列表（User 对象，包含静态信息，如 task 参数、local_cpu 等）
      X: offloader 用户集合（用户 id 的集合或列表）
      resource_setting: 资源设置对象，包含 f_max, B_max, p_E, p_N 等（这里将用 p_E, p_N 更新）
      epsilon: 外部稳定性阈值
      S_i: 信道参数（通常设为1或根据用户情况调整）
      
    返回：
      True 如果 (p_E, p_N) 属于 P_X，否则返回 False。
    """

    if is_price_pre_feasible(users, provider, X, p_E, p_N) is False:
      if verbose: print("is_price_feasible: price (",p_E, ",", p_N, ")  is not pre feasible")
      return False

    # 提取 offloader 用户列表（即 user.user_id in X）
    users_X = [user for user in users if user.user_id in X]

    # 调用 ORA 求解器，求解 offloader 用户在当前价格下的最优资源分配
    opt_f, opt_B, opt_cost = ora_solver(users_X, provider, p_E, p_N)
    if opt_f is None or opt_cost is None:
        # ORA 无法求解或不可行
        if verbose: print("is_price_feasible: unable to solve ORA(X)")
        return False

    remain_f = max(provider.f_max - np.sum(opt_f), 0)
    remain_B = max(provider.B_max - np.sum(opt_B), 0)

    # 对于不在 X 中的用户，使用闭式估计检查外部稳定性条件：
    # 设定 f_hat = sqrt(alpha*d / p_E), B_hat = sqrt(alpha*b / (S_i*p_N))
    # 然后 offloading cost = alpha*(b/(B_hat*S_i) + d/(f_hat)) + p_E*f_hat + p_N*B_hat
    for user in users:
        if user.user_id not in X:
          if user.offload_best_response(p_E, p_N, remain_f, remain_B) == 1:
            if verbose: print("is_price_feasible: not externally stable")
            return False

    return True

def is_price_pre_feasible(users, provider, X, p_E, p_N, verbose=False):
    """
    检查给定的价格组合是否属于预可行域 P_X^{pre}.
    如果价格导致根号内出现负值，则返回False。

    参数：
    - p_E, p_N: 当前价格
    - users: 用户列表
    - X: offloader 用户id集合
    - provider: 资源提供商参数 (包含f_max, B_max等)
    - S_i: 信道参数，默认为1

    返回：
    - True 如果价格在P_X^{pre}内，否则返回False
    """
    users_X = [user for user in users if user.user_id in X]

    for user in users_X:
      if user.C_hat_e_i(p_E,p_N) >= user.cost_local()+1e-8:
        if verbose: print("is_price_pre_feasible: Not pre feasible. C_hat_e",user.user_id," > C_lcoal_",user.user_id)
        if verbose: print(f"{user.C_hat_e_i(p_E,p_N)}>{user.cost_local()}")
        return False
    
    f_thres_sum = np.sum([user.f_thres(p_E,p_N) for user in users_X])
    B_thres_sum = np.sum([user.B_thres(p_E,p_N) for user in users_X])
    if f_thres_sum >= provider.f_max+1e-8:
      if verbose: print("is_price_pre_feasible: sum of f_thres > f_max")
      return False
    if B_thres_sum >= provider.B_max+1e-8:
      if verbose: print("is_price_pre_feasible: sum of B_thres > B_max")
      return False
    return True

def ora_solver(offloaders, provider, p_E, p_N, verbose=False):
    """
    求解给定 offloader 用户集合 X 下的最优资源分配问题 ORA(X)
    
    输入：
      offloaders: offloader 用户列表，每个用户具有属性：
             user.task.alpha, user.task.d, user.task.b, user.local_cpu
      resource_setting: 资源设置对象，包含属性：
             f_max, B_max, p_E, p_N
      S_i: 信道转换参数（默认为1.0，可根据具体情况设置）
    
    输出：
      optimal_f: 数组，最优的计算资源分配（对应每个用户）
      optimal_B: 数组，最优的带宽资源分配（对应每个用户）
      optimal_cost: 优化问题的目标函数值（社会成本）
    """

    n = len(offloaders)
    if n == 0: 
      print("ora_solver: offloaders is empty")
      return None, None, None

    # 为每个用户计算闭式上界及下界
    f_hat_list, B_hat_list = [], []
    f_thresh_list, B_thresh_list = [], []
    local_cost_list = []  # 用户本地计算成本 C_l = alpha*(d / local_cpu)
    for user in offloaders:
        alpha = user.task.alpha
        d = user.task.d
        b = user.task.b
        # 闭式上界：f_hat = sqrt(alpha * d / p_E)，B_hat = sqrt(alpha * b / (S_i * p_N))
        f_hat = np.sqrt(alpha * d / p_E)
        B_hat = np.sqrt(alpha * b / (user.S_i * p_N))
        
        C_l_i = alpha * (d / user.local_cpu)
        C_hat_eb_i = 2 * np.sqrt(alpha * b * p_N / user.S_i)
        C_hat_ef_i = 2 * np.sqrt(alpha * d * p_E)
        # 闭式下界
        if C_l_i >= C_hat_eb_i + C_hat_ef_i:
          f_thresh = ((C_l_i - C_hat_eb_i) - np.sqrt((C_l_i - C_hat_eb_i)**2 - C_hat_ef_i**2)) / (2*p_E)
          B_thresh = ((C_l_i - C_hat_ef_i) - np.sqrt((C_l_i - C_hat_ef_i)**2 - C_hat_eb_i**2)) / (2*p_N)
        else:
          if verbose: print(f"ora_solver: Problem infeasible, C_l_{user.user_id}={C_l_i}<C_hat_eb_{user.user_id}+C_hat_ef_{user.user_id}={C_hat_eb_i}+{C_hat_ef_i}={C_hat_eb_i+C_hat_ef_i}")
          return None, None, None
        
        f_hat_list.append(f_hat)
        B_hat_list.append(B_hat)
        f_thresh_list.append(f_thresh)
        B_thresh_list.append(B_thresh)
        
        # 本地成本：C_l = alpha * (d / local_cpu)
        local_cost_list.append(alpha * (d / user.local_cpu))
    
    f_hat_arr = np.array(f_hat_list)
    B_hat_arr = np.array(B_hat_list)
    f_thresh_arr = np.array(f_thresh_list)
    B_thresh_arr = np.array(B_thresh_list)
    local_cost_arr = np.array(local_cost_list)

    # 决策变量向量 x: 前 n 个为 f_i, 后 n 个为 B_i
    # 初始猜测：取中间值
    x0 = np.concatenate(((f_hat_arr + f_thresh_arr) / 2, (B_hat_arr + B_thresh_arr) / 2))
    
    # 定义目标函数：总卸载成本
    def objective(x):
        f_vals = x[:n]
        B_vals = x[n:]
        total_cost = 0.0
        for i, user in enumerate(offloaders):
            alpha = user.task.alpha
            d = user.task.d
            b = user.task.b
            # 成本函数：C_e = alpha*(b/(B_i*S_i) + d/(f_i)) + p_E*f_i + p_N*B_i
            cost_i = alpha * (b/(B_vals[i]*user.S_i) + d/(f_vals[i])) + p_E * f_vals[i] + p_N * B_vals[i]
            total_cost += cost_i
        return total_cost

    # 定义约束
    cons = []
    # 全局资源约束：sum(f) <= f_max, sum(B) <= B_max
    cons.append({'type': 'ineq', 'fun': lambda x: provider.f_max - np.sum(x[:n])})
    cons.append({'type': 'ineq', 'fun': lambda x: provider.B_max - np.sum(x[n:])})

    # 每个用户卸载成本不超过本地计算成本： C_e_i(f_i,B_i) <= C_l_i
    def cost_constraint(x, i):
        f_i = x[i]
        B_i = x[n+i]
        alpha = offloaders[i].task.alpha
        d = offloaders[i].task.d
        b = offloaders[i].task.b
        C_e = alpha * (b/(B_i * user.S_i) + d/(f_i)) + p_E * f_i + p_N * B_i
        return local_cost_arr[i] - C_e  # 要求 >= 0

    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: cost_constraint(x, i)})

    # 使用 SLSQP 求解，并设置较高的迭代上限和较高的精度
    res = minimize(objective, x0, method='SLSQP',
               bounds=[(max(f_thresh_arr[i], 1e-6), f_hat_arr[i]) for i in range(n)] +
                      [(max(B_thresh_arr[i], 1e-6), B_hat_arr[i]) for i in range(n)],
               constraints=cons,
               options={'disp': False, 'ftol': 1e-4, 'maxiter': 1000})

    # res = minimize(objective, x0, method='trust-constr',
    #            bounds=[(max(f_thresh_arr[i], 1e-6), f_hat_arr[i]) for i in range(n)] +
    #                   [(max(B_thresh_arr[i], 1e-6), B_hat_arr[i]) for i in range(n)],
    #            constraints=cons,
    #            options={'verbose': 0, 'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 1000})

    # res = minimize(objective, x0, method='trust-constr',
    #            bounds=[(max(f_thresh_arr[i], 1e-6), f_hat_arr[i]) for i in range(n)] +
    #                   [(max(B_thresh_arr[i], 1e-6), B_hat_arr[i]) for i in range(n)],
    #            constraints=cons,
    #            options={'verbose': 0, 'gtol': 1e-6, 'xtol': 1e-6, 'maxiter': 2000})

    
    if not res.success:
        print("Optimization failed:", res.message)
        return None, None, None

    x_opt = res.x
    optimal_f = x_opt[:n]
    optimal_B = x_opt[n:]
    optimal_cost = res.fun
    return optimal_f, optimal_B, optimal_cost
