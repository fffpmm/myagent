import ast
import datetime
import threading
from dataclasses import asdict, dataclass
import json
import random
import re
import time
import yaml
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path



load_dotenv(override=True)
#给s3用 当前的工作目录
WORKDIR = Path.cwd()
#给s4用
CURRENT_TODOS:list[dict]=[]

SKILLS_DIR = Path(__file__).parent / "skills"

#这个__main__中配合 build_skill_system和build_memory_system使用
#SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain. "

#只给主agent进行调用 subagnet就不给skill了
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)

# 给一个压缩tool结果函数使用的地址 让那个过长的tool_content放入这个目录
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

# 压缩的工具调用结果放到该目录当中
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

#存放各种memory文件的地方
MEMORY_DIR = WORKDIR / ".memory"; MEMORY_DIR.mkdir(exist_ok=True)

#存放所有memory信息的索引文件
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

# 默认模型
PRIMARY_MODEL = "deepseek-v4-pro"

#备用模型 error_recovery会用
FALLBACK_MODEL = "deepseek-v4"

"""   【该agent在运行功能时会产生的系统目录与文件】
.task_outputs / tool-results/<tool_use_id>.txt
.transcripts / transcript_<timestamp>.jsonl
skills/<skill_name>和SKILL.md
"""

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

#==========定时操作系统===================================================================================================================
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"

#定时任务的数据结构
@dataclass
class CronJob:
    id: str
    #cron表达式 规定任务触发周期
    cron: str        # "0 9 * * *"
    #任务触发时 要注入大模型的提示内容
    prompt: str      # message to inject when fired
    #True为重复执行
    recurring: bool  # True = recurring, False = one-shot
    #True为持久化任务
    durable: bool    # True = persist to disk

#===全局容器与锁变量===

#键是任务id，值是cornjob实例  存放所有已注册的定时任务便于id快速查找
scheduled_jobs: dict[str, CronJob] = {}
#待执行任务队列 扫描到已触发条件的任务 先放进队列 后续交给agent处理
cron_queue: list[CronJob] = []
#定时任务专用锁
cron_lock = threading.Lock()
#agent业务逻辑专用锁
agent_lock = threading.Lock()
#记录上一次任务触发时间
_last_fired: dict[str, str] = {}  # job_id → "YYYY-MM-DD HH:MM"


#====关于cron匹配与校验======

def _cron_field_matches(field: str, value: int) -> bool:
    """
    校验cron单字段是否匹配当前时间数值

    field：cron单格的规则字符串
    value：当前时间时间对应的数字
    """
    # 每个if都是一种cron的通配符类型
    if field == "*":
        return True
    if field.startswith("*/"):
        #因为*/数字  这个数字就代表一个步长  所以直接提取然后开除
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        #因为 ，代表有多种时间点 所以用逗号进行切割 拿到每一种再重新判断
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        #之所用int()转换成数字 因为虽然lo和hi接受字符串但需要比较大小刚好这俩也是数字
        return int(lo) <= value <= int(hi)
    return value == int(field)

def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """把完整五段corn表达式，传入datetime时间对象，判断是否触发定时任务"""
   
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    # 星期换算 python里周一=0 周日=6 所以做数值转换 由于是%触发所以必能拿到周几
    dow_val = (dt.weekday() + 1) % 7  

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    # Minute, hour, month 必须同时满足
    if not (m and h and month_ok):
        return False
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    
    # 日期和星期任意一个是* 就直接可触发
    if dom_unconstrained and dow_unconstrained:
        return True
    # 如果人一个是* 只需另一个满足匹配上了时间也可以触发任务
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    #若都设置了约束 日期命中或者星期命中 满足其一就触发任务
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """
    用户新建定时任务时提前检查cron写法有没有语法数值错误

    field：cron单格的规则字符串
    lo：cron单格最小值
    hi：cron单格最大值

    返回：None表示合法  返回字符串表示校验不通过，字符串是具体报错信息
    """
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        # 判断是否是纯数字 小数点什么符号都不能有
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err: return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None

def validate_cron(cron_expr: str) -> str | None:
    """校验五段cron表达式是否合法"""
    
    # 把完整五段corn表达式 分割成五段
    fields = cron_expr.strip().split()
    
    # 长度必须为5
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    
    # 每个字段的数值范围
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    # 把字段内容，取值上下限，字段名称配对 调用上一节的_validate_cron_field校验单段 只要任意一段出错 就返回错误
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


#===对于全局字典的cronjob值操作====

def save_durable_jobs():
    """把所有需要持久化的定时任务对象写入json文件"""
    # 遍历所有定时任务（遍历全局子弹） ，筛选durable=True的任务，asdict把dataclass的cornjob对象转为字典 方便json序列化
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))



def load_durable_jobs():
    """程序启动时加载持久化定时任务"""
    # 判断持久化json文件是否存在
    if not DURABLE_PATH.exists():
        return
    
    try:
        # 读出来的是json字符串是无法转化成字典 所以json.loads()把json字符串或列表转为字典或列表 可以进行正常python操作
        jobs = json.loads(DURABLE_PATH.read_text())
        # 遍历每一条任务字典
        for j in jobs:
            # 构造 CronJob对象
            job = CronJob(**j)
            # 校验cron 校验通过就存入全局字典 scheduled_jobs里否则跳过该任务
            err = validate_cron(job.cron)
            if err:
                print(f" [cron] skipping invalid job {job.id}: {err}")
                continue
            scheduled_jobs[job.id] = job

        # 与程序无关 只是打印再终端 将上面操作结果
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f" [cron] loaded {len(valid)} durable job(s)")
    except Exception:
        pass

def create_schedule_job(cron: str, prompt: str, recurring: bool = True,
            durable: bool = True) -> CronJob | str:
    """
    新增注册一条定时任务
    
    返回一个返回值：CronJob对象：任务注册成功
    字符串：任务注册失败
    """
    
    # 校验传入的cron表达式格式有误返回错误字符串
    err = validate_cron(cron)
    if err:
        return err
    # 生成唯一任务id 构造cronjob对象
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
   
    # 加入cron_lock锁 把任务写入全局字典 scheduled_jobs 防止多线程并发写入字典错误
    with cron_lock:
        scheduled_jobs[job.id] = job
    
    # 如果开启持久化了 就立刻调用save_durable_jobs把最新任务和之前存的写入文件中    
    if durable:
        save_durable_jobs()
    print(f" [cron register] {job.id} '{cron}' → {prompt[:40]}")
    return job


def cancel_job(job_id: str) -> str:
    """取消/删除已有定时任务"""
    
    #防止多线程并发操作
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    
    if not job:
        return f"Job {job_id} not found"
    
    # 删除定时任务是在全局字典里的操作 所以要判断任务是否持久化再重置下文件中的定时任务让内存与文件一致
    if job.durable:
        save_durable_jobs()
    print(f" [cron cancel] {job_id}")
    return f"Cancelled {job_id}"



def cron_scheduler_loop():
    """
    定时任务调度后台循环

    每秒扫描扫描所有定时任务，判断是否需要执行 如果任务没有触发的话就放入待执行列表并将_last_fired字典里该任务对应的时间
    """
    while True:
        # 每隔1秒休眠一次 降低cpu占用
        time.sleep(1)
        
        # 获取当前时间now 生成minute_marker字符串
        now = datetime.now() 
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
       
        #加入cron_lock锁 遍历所有定时任务 
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            print(f" [cron fire] {job.id} → "
                                  f"{job.prompt[:40]}")
                        # 如果是一次性任务出发后直接从内存任务中删除
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f" [cron error] {job.id}: {e}")


def consume_cron_queue() -> list[CronJob]:
    """清空已触发的任务队列，给agent调用
    
    返回已触发的任务列表"""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def has_cron_queue() -> bool:
    """判断是否有任务队列"""
    with cron_lock:
        return bool(cron_queue)


load_durable_jobs()
#调度循环放入守护子线程  主线程退出时这个调度线程自动结束
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print(" [[cron]  thread started")



# ── Cron 工具函数 ──

def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """创建定时任务的工具入口 创建成功返回文本告知成功"""
    result = create_schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    """查询全部已有定时任务"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"
        dur = "durable" if j.durable else "session"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)

def run_cancel_cron(job_id: str) -> str:
    """取消指定定时任务的工具入口"""
    return cancel_job(job_id)


#==========持久化任务系统==============================================================================================================
#就像排火车一个个完成 不能越过一个完成下一个

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    id: int
    #相当于简介
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # Agent name (multi-agent scenarios)
    #列表里面存放的都是前置任务的task_id
    blockedBy: list[str] 
#创建任务文件路径
def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"
#实例化任务对象并存入json文件 【第三个函数使用第一个函数功能第二个函数使用第三个函数功能达到闭环】
def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """是每个任务里有个blockedby待做项"""
    task = Task(
        # 生成任务id time.time()生成时间戳 从1970到现在总秒数带小数所以int()
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task

#  下面几乎都能创建文件
#将task对象内容转成字典再转成json字符串存入json文件 为上面函数服务
def save_task(task: Task):
    """
    将task对象转成字典并保存为json文件 会顺便创建文件
    将任务对象内容保存在文件中
    """
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))

#加载单个任务对象 【根据id从任务文件加载任务对象】
def load_task(task_id: str) -> Task:
    """返回task对象  会顺便创建文件"""
    return Task(**json.loads(_task_path(task_id).read_text()))

#加载所有任务对象
def list_tasks() -> list[Task]:
    """返回task对象列表"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]

#获取单个任务对象详情 返回的json字符串
def get_task(task_id: str) -> str:
    """返回该指定task_id的任务详情 完整的一个json字符串"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)

#【判断任务能否启动】
def can_start(task_id: str) -> bool:
    """
    校验当前任务所有前置依赖(blockedBy里任务)是否完成
    1.读取当前任务
    2.遍历每一个依赖任务task_id 
        如果以来任务json文件不存在当前任务不能启动返回false
        如果依赖任务状态不是completed不能启动返回false
    3.所有依赖任务完成返回true代表任务可以执行
    """
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True

#【申请，认领任务】
def claim_task(task_id: str, owner: str = "agent") -> str:
    """
    认领任务
    1.读取当前任务
    2.校验任务状态：只有pending才能认领，已完成就拒绝认领
    3.调用can_start再次校验依赖,依赖没就绪就返回阻塞信息不让认领 相当于这个函数里返回false就代表了不让认领 如已完成和路径不存在   
    4.校验通过:给任务绑定owner(哪个agent接手),状态改成in_progress,保存任务
    5.返回认领成功信息

    通俗易懂:agent接活,确认活没人做,前置条件满足后,把活揽到自己身上开始做
    """

    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)

    print("[claim] {task.subject} → in_progress (owner: {owner})")
    return f"Claimed {task.id} ({task.subject})"

#【完成任务,解锁下游依赖函数】
def complete_task(task_id: str) -> str:
    """
    1.读取当前任务
    2.校验状态：只有in_progress才能完成
    3.完成任务:修改任务状态为completed,保存任务
    4.遍历全部任务:筛选状态是pending且依赖包含刚完成的任务,现在依赖全部满足的下游任务,存入unblcked列表

    通俗理解:做完手头任务后,系统检查那些卡在这个任务后面的工作现在可以开工了,把这些可开工的任务汇总通知调度器
    """
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)

    #都是用来拿到后面所有任务弄成字符串通知可以开始了
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"[complete] {task.subject} ✓")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"[unblocked] {', '.join(unblocked)}")
    return msg


#==== 定义task工具 =====

#<====== 创建任务 ======>
def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f" [create] {task.subject}{deps}")
    return f"Created {task.id}: {task.subject}{deps}"

#<====== 列出所有任务 ======>
def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        #将每个任务转成一个字符串(内有blockedby owner suject id icon)放进列表中再转成字符串
        icon = {"pending": "○", "in_progress": "●","completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)

#<======= 获取单个任务详情 ======>
def run_get_task(task_id: str) -> str:
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"

#<======== 认领任务 ======>
def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")

#<======== 完成任务 ======>
def run_complete_task(task_id: str) -> str:
    return complete_task(task_id)





#==========模型调用错误处理机制====================================================================================================
#升级后最大输出token上限
ESCALATED_MAX_TOKENS = 64000
#常规请求默认最大输出token
DEFAULT_MAX_TOKENS = 8000
#最大重试次数
MAX_RECOVERY_RETRIES = 3
#单次接口请求重试次数
MAX_RETRIES = 10
#基础等待毫秒
BASE_DELAY_MS = 500
#连续529错误阀值
MAX_CONSECUTIVE_529 = 3
#触发token上限后的续写提示
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

#状态存储  【全局记录本轮循环的重试状态】
class RecoveryState:
    """Track recovery attempts across the loop."""
    def __init__(self):
        #是否切换过备用模型
        self.has_escalated = False
        #累计重试总次数
        self.recovery_count = 0
        #连续的529错误次数
        self.consecutive_529 = 0
        #是否执行上下文压缩降级策略
        self.has_attempted_reactive_compact = False
        #当前使用的模型
        self.current_model = PRIMARY_MODEL

#计算重试等待时间(指数退避+随即抖动)
def retry_delay(attempt, retry_after=None):
    """attempt参数就是给with_retry()函数调用的参数
    返回值是每次重试等待时间"""
    if retry_after:
        return retry_after
    # 2**attempt是指数退避 等待时间每次成倍变长 上限记忆是32000毫秒也就是32秒
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    #在这个区间生成随机浮点数
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def with_retry(fn, state: RecoveryState):
    """大模型API请求的容错重试包装函数专门处理429请求限流和529服务商服务器过载
    搭配指数退避延时和备用模型切换，遇到不可临时修复错误向上抛出交给外层代码处理
    fn：模型调用函数
    state：状态存储对象"""

    # 模型请求成功 清空连续529 返回模型结果
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as e:
            #捕获异常类名 报错文本转小写方便匹配429请求频次超限限流和529关键字服务商服务器过载
            name = type(e).__name__
            msg = str(e).lower()

            # 判断是否429限流错误  是的话就进行重试
            if "ratelimit" in name.lower() or "429" in msg:
                delay = retry_delay(attempt)
                print(f"[429 rate limit] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s")
                time.sleep(delay)
                continue

            # 判断是否529过载错误 如果既不是429或529错误直接报错
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                #判断连续过载次数是否达到阈值 超过就切换模型 如果没有备用模型就继续重试
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL:
                        state.current_model = FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(f"{MAX_CONSECUTIVE_529}]"
                              f" switching to {FALLBACK_MODEL}")
                    else:
                        state.consecutive_529 = 0
                        print(f"{MAX_CONSECUTIVE_529}]"
                              f" no FALLBACK_MODEL_ID configured, continuing retry")
                # 延迟个一些时间再次重试
                delay = retry_delay(attempt)
                print(f" [529 overloaded] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s")
                time.sleep(delay)
                continue

            raise
    # 超过最大重试次数 还是无法解决 抛出异常
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """捕获异常是否是上下文超出模型窗口上限的错误"""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "prompt_is_too_long" in msg
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


def error_recovey_energency_compact(messages: list) -> list:
    """应急精简对话历史，解决上下文超长问题"""
    print(" [reactive compact] trimming to last 5 messages")
    tail = messages[-5:]
    return [{"role": "user",
             "content": "[Reactive compact] Earlier conversation trimmed. "
                        "Continue from where you left off."}, *tail]




#===========记忆功能====================================================================================================
MEMORY_TYPES = ["user", "feedback", "project", "reference"]

def extract_text_memory(response: str) -> str:
    return response.choices[0].message.content


#用来处理文件的数据 提取数据到变量中
def _parse_frontmatter_memory(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    # parts是列表
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    # 将frontmatter拿到也就是两个---中间的内容拿到 开始分割
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    # 一个是frontmatter 一个是完整content
    return meta, parts[2].strip()

#在MEMORY_DIR也就是当前目录的.memory下构建 {name}.md文件 返回该path路径  供其他函数能够拿到里面的数据
def write_memory_file(name: str, mem_type: str, description: str, body: str):
    # 构建记忆文件在MEMORY_DIR下
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n")
    _rebuild_index()
    return filepath


#在MEMORY_DIR下构建MEMORY.md文件
def _rebuild_index():
    # MEMORY_INDEX就是构建MEMORY.md文件索引的  这里先用lines接受每个上个函数构建的{name}.md文件取得name等信息传进MEMOIRY.md文件中
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter_memory(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")



# 拿取MEMORY.md文件里的内容并返回
def read_memory_index() -> str:
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""

#拿取 该filename记忆文件的内容 不是MEMORY.md文件
def read_memory_file(filename: str) -> str | None:
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


#列出所有记忆文件除了MEMORY.md 并拿到他的数据构成一个个字典放入列表
def list_memory_files_fetch_sourcelist() -> list[dict]:
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter_memory(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result

#<===智能记忆检索函数===>  【根据用户对话筛选出和聊天相关记忆的文件名】  ai精准筛选和关键字兜底匹配
def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    #确认该函数拿到了各个memory资源列表内是一个个字典
    list_source = list_memory_files_fetch_sourcelist()
    if not list_source:
        return []

    #将最近三条用户的提问提取出来 放进recent_texts列表中最后转化成recent字符串
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]

    #判断recent字符串存在吗
    if not recent.strip():
        return []

    # 这里是拿到每个memory的id以及name和描述 这里id有用到时候ai返回就是指认idmemory
    catalog_lines = []
    for i, f in enumerate(list_source):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)

    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
    )

    # 上面第一个拿到前三条用户的提问消息构成recent 第二个拿到所有记忆的name和描述
    system = (
              f"Recent conversation:\n{recent}\n\n"
              f"Memory catalog:\n{catalog}")

    try:
        # <===AI智能检索===>
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
        )

        
        # 该方法定义在subagent那里 用于拿到response的普通回答消息content不要tool_calls
        text = extract_text_memory(response).strip()
        # 由于给模型的要求返回数字id列表 match就是一个对象 里面是想要记忆的id列表 匹配的是[0,2]
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            # 将match对象转成字符串 再转化成列表
            indices = json.loads(match.group())
            # 拿到列表后开始检验列表里id是否满足  满足的话就换成名字存入列表
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(list_source):
                    selected.append(list_source[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass

        # <===关键字模糊匹配兜底===>
        keywords = [w.lower() for w in recent.split() if len(w) > 3]
        selected = []
        for f in list_source:
            text = (f["name"] + " " + f["description"]).lower()
            if any(kw in text for kw in keywords):
                selected.append(f["filename"])
                if len(selected) >= max_items:
                    break
        return selected

#将上面智能检索的记忆内容放进列表中变成字符串  【通过智能检索出来的文件名去拿到该文件的内容最后将内容放进列表转化成字符串】
def load_memories(messages: list) -> str:
     #这个函数返回就是memory文件名字列表
     selected_files = select_relevant_memories(messages)
     if not selected_files:
        return ""

     parts = ["<relevant_memories>"]
     for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
     parts.append("</relevant_memories>")
     return "\n\n".join(parts)

#<===智能记忆提取===>  【将最近十条消息和以前的消息记忆文件提取成字符串合起来让ai返回规定的列表内套字典的格式然后提取出内部的数据然后创建出一个个字典对应的一个个文件】 
def extract_memories(messages: list):
    #<===第一步===>
    #这里就是提取对话列表内容从倒数第10到最后一个的content连成一个"字符串"
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    # 判断提取出来的对话字符串是否存在
    if not dialogue.strip():
        return

    #<===第二步===>
    # 拿取所有记忆文件的详细列表转化成一行相当于一个文件的基本内容的"字符串"
    existing = list_memory_files_fetch_sourcelist()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    system = (
        "你是一个记忆提取助手。从对话中提取需要长期记住的新信息。\n\n"
        "每条记忆包含四个字段：\n"
        "1. name：简短横杠分隔命名，示例：user-preference-tabs\n"
        "2. type：仅能选用4种值：user / feedback / project / reference\n"
        "3. description：单行简短摘要，用于快速检索记忆\n"
        "4. body：完整详细内容，使用Markdown格式书写\n\n"
        "如果没有新增信息，或是内容已经存在于已有记忆中，直接返回空数组 []\n\n"
        f"已有记忆：\n{existing_desc}\n\n"
        f"对话内容：\n{dialogue[:4000]}"
    )

    prompt = "请根据上述对话内容，提取需要新增的记忆，返回 JSON 数组。如果没有新信息则返回 []。"

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
        )

        # 拿到response的普通回答消息content
        text = extract_text_memory(response).strip()
        # 防止ai返回的还有文本 这个可以剔除文本只拿[]
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return

        count = 0
        # 提取ai被要求的json格式 并拿到里面的各个内容 将其创建一个memory文件
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n[Memory: extracted {count} new memories]")
    except Exception:
        pass

# 最大允许提取memory文件的数量
CONSOLIDATE_THRESHOLD = 10


#<===智能整合老记忆文件加入新记忆文件===> 【调整当前memory文件将里面所有数据提取成字符串发给模型过时的和合并重复的重新变成一个列表内含一个个字典删去老的文件除了MEMORY创建新的文件】
def consolidate_memories():
    files = list_memory_files_fetch_sourcelist()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return
    
    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    system_prompt = (
        "整合以下记忆文件。规则：\n"
        "1. 合并重复的记忆为一条\n"
        "2. 删除过时或矛盾的内容\n"
        "3. 总数控制在30条以内\n"
        "4. 优先保留重要的用户偏好\n"
        "返回 JSON 数组，每项：{name, type, description, body}。"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"以下是记忆文件：\n\n{catalog[:16000]}"},
            ],
            max_tokens=3000,
        )
        text = extract_text_memory(response).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # 去除老的文件除了MEMORY.md
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
        print(f"[记忆：合并 {len(files)} → {len(items)} 条记忆]")
    except Exception:
        pass


def build_memory_system() -> str:
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )






#===========SKILL加载工具函数===========================================================================================

#解析文件中yaml配置信息  meta（part(1）拿到就是上下---中间的内容这个内容就是简介 part（2）相当于详细内容
def _parse_frontmatter(text):
    if not text.startswith("---"):
        return {},text
    part = text.split("---",2)
    if len(part) < 3:
        return {},text
    try:
        meta = yaml.safe_load(part[1])
    except yaml.YAMLError:
        meta = {}
    return meta,part[2].strip()

# 该变量就是将收集的简要信息存在这里面  到时候交给systemprompt
SKILL_REGISTRY: dict[str, dict] = {}
#扫描所有的skills目录下的SKILL的文件 将他的部分内容作为简介拿出构成字典到时候放入模型prompt让模型知道有那些skill
def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        #这个SKILL需要提出注入这个技能这个里/SKILL类似于已经有这个技能了那个变量接收这个路径而已
        manifest = d / "SKILL.md"
        if manifest.exists():
            # 拿到SKILL.md文件里所有内容 使用_parse_frontmatter函数进行分割取到各个信息
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            # name 方便找到想要使用的skill description方便让模型知道这个skill干啥的 content 方便想要使用时拿到skill的内容
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


_scan_skills()

#列出所有技能 将含所有的技能简介信息转化返回一个大的介绍技能字符串  以便放入systemprompt字符串
def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

# 建立systemprompt字符串
def build_skill_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )


#只给主agent进行调用 subagnet就不给skill了
SKILL_SYSTEM = build_skill_system()

#<====== 工具函数 ======>  给主agent进行调用 真正让模型调用的工具函数 想要使用时传来的技能名就能拿到内容
def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


#============核心功能工具函数==================================================================================================
def run_bash(command:str):
    # dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # if any(a in command for a in dangerous):
    #     return "error:出现了越级的命令"
    try:
        # shell为ture是传完整字符串即可，不然需要把命令拆开变成列表，cwd代表命令的工作目录，capture_output相当于stderr和stdout正常消息和报错消息打印到终端
        r=subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out=(r.stdout+r.stderr).strip()
        return out[:50000] if out else "没有out"
    except subprocess.TimeoutError:
        return "error:超时120"

def safe_path(p:str)->str:
    path = (WORKDIR/p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径跳出了工作目录: {p}")
    return path


def run_read(path:str,limit:int|None=None):
    try:
        lines=safe_path(path).read_text().splitlines()
        if limit and limit<len(lines):
            lines = lines[:limit] + [f"还有{len(lines)-limit}行未显示"]
        return "\n".join(lines)
    except Exception as e:
        return f"error:读取文件错误{e}"

def run_write(path:str, content:str):
    try:
        safepath = safe_path(path)
        safepath.parent.mkdir(parents=True,exist_ok=True)
        safepath.write_text(content)
        return f"写入{len(content)}行到{path}中"
    except Exception as e:
        return f"error:写入文件错误{e}"

def run_edit(path:str,old_content:str,new_content:str):
    try:
        safepath = safe_path(path)
        old_text=safepath.read_text()
        if old_content not in old_text:
            return "error:内容不一致"
        safepath.write_text(old_text.replace(old_content,new_content,1))
        return "消息替换完成"
    except Exception as e:
        return f"error:编辑文件错误{e}"

#文件检索工具，让大模型通过通配符搜索项目工作目录WORKDIR的文件
def run_glob(patten:str):
    try:
         result = []
         for match in WORKDIR.glob(patten):
             if (WORKDIR/match).resolve().is_relative_to(WORKDIR):
                result.append(match)
         return "\n".join(result) if result else "(no matches)"
    except Exception as e:
        return f"Error: {e}"



#========聚焦任务目标工具函数（该工具不进行外部操作只做提醒和打印在终端）======================================================================
# TodoWrite 是当前任务的执行清单，保存在会话内存中

# 对todos的消息类型进行检测 只要列表里是字典的   todos这个参数也是像其它工具函数一样的由大模型定义给它
def _normalize_todos(todos):
    if isinstance(todos,str):
        try:
            todos =json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError,ValueError):
                return None,"error:todos 参数格式错误"
    if not isinstance(todos,list):
        return None,"error:todos 列表格式错误"
    for i,g in enumerate(todos):
        if not isinstance(g,dict):
            return None,f"error:todos[{i}]的必须是一个字典"
        if "content" not in g or "status" not in g:
            return None,f"error:todos[{i}]可能缺少content或status"
        if g["status"] not in ("pending","in_progress","completed"):
            return None,f"error:todos[{i}]未设定状态"
    return todos,None

#<======工具函数  ====>  为状态添加上符号并打印出每个任务的状态  todos这个参数也是像其它工具函数一样的由大模型定义给它
def run_todo_write(todos:list):
    global CURRENT_TODOS
    todos_ture,error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos_ture
    lines=["\n\n##当前的任务"]
    icon_map ={
        "pending":"",
        "in_progress":"▶",
        "completed":"✔"
    }
    for todo_item in CURRENT_TODOS:
        status_icon = icon_map[todo_item["status"]]
        task_text = todo_item["content"]
        lines.append(f"{status_icon} {task_text}")
    print("\n".join(lines))
    return f"更新了{len(CURRENT_TODOS)}个任务"


# =========子agent=====================================================================================================

# 子agent工具调用
SUB_TOOLS = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "运行一个命令去工作",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "设置一个命令"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "read",
            "description": "读取一个文件的内容",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "limit":{
                        "type":"integer",
                        "description":"限制读取的行数"
                    }
                }
            }
        }
    },{


        "type": "function",
        "function": {
            "name": "write",
            "description": "写入一个文件的内容",
            "parameters": {
                "type": "object",
                "required": ["path","content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content":{
                        "type":"string",
                        "description":"写入的内容"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "edit",
            "description": "编辑一个文件",
            "parameters": {
                "type": "object",
                "required": ["path","old_content","new_content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "old_content":{
                        "type":"string",
                        "description": "要被替换的内容"
                    },
                    "new_content":{
                        "type":"string",
                        "description": "新的内容"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "glob",
            "description": "检索文件",
            "parameters": {
                "type": "object",
                "required": ["patten"],
                "properties": {
                    "patten": {
                        "type": "string",
                        "description": "检索的通配符"
                    }
                }
            }
        }
    },
]


SUB_TOOL_HANDERS = {
    "bash": run_bash,
    "read":run_read,
    "write":run_write,
    "edit":run_edit,
    "glob":run_glob,
}

# block传来的将是messages也就是消息列表 【将消息列表转化成字符串】
def extract_text_subagent(block):
    return "\n".join(b.content for b in block if not b.tool_calls and b.content)

#<====== 工具函数 ======>
def spawn_agent(description):
    print("[SPAWN AGENT]")
    messages =[
        {"role": "system", "content": SUB_SYSTEM},
        {"role": "user", "content": description}
    ]

    for _ in range(30):
        response = client.chat.completions.create(
            model = "deepseek-v4-pro",
            messages = messages,
            tools = SUB_TOOLS,
            max_tokens=8000
        )
        meg = {"role":"assistant","content":response.choices[0].message.content}
        if response.choices[0].message.tool_calls:
            # 主agent用的是for循环一个个拿字典的key和value 这里直接model_dump
            meg["tool_calls"] = response.choices[0].message.tool_calls.model_dump()
        messages.append(meg)

        if response.choices[0].finish_reason != "tool_calls":
            trigger_hook("AFTER_AGENT",messages)
            break


        for tc in response.choices[0].message.tool_calls:
            if not tc:
                continue
            blocked=trigger_hook("BEFORE_TOOL", tc)
            arg=json.loads(tc.args)
            output=SUB_TOOL_HANDERS[tc.name](**arg)
            print(f"[sub] {tc.name}: {str(output)[:100]}")
            messages.append({"role":"tool","content":output})

    result=extract_text_subagent(messages)
    if not result:
        for msg in reversed(messages):
            if msg["role"]=="assistant":
                result=msg["content"]
                if result:
                    break
        if not result:
            return "子agent在30轮对话后结束了"
    print(f"[SUB AGENT 结束———]")
    return result


# =========五层上下文消息压缩===============================================================================================
#前三层如hook放入agent_loop里自动判断触发第四层是工具函数由模型调用
# $$注意在l4和l5的时候会将system消息也给压缩了 所以要注意补上 已经在agent__loop补上了


#上下文消息最终限制 若多余该值会触发紧急压缩
CONTEXT_LIMIT = 50000

#作为指针压缩这个值的负数的前面tool结果
KEEP_RECENT = 3

#判断一个tool的结果是否满足这个值 若满足tool的结果则会被放进文件里
PERSIST_THRESHOLD = 30000

#放进agent_loop里所有消息大小是否触发紧急压缩
def estimate_size(msgs): return len(str(msgs))

# 判断是否是调用工具的msg
def _message_has_tool_calls(msg):
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return True

#判断是否是调用工具的
def _is_tool_result_message(msg):
    if msg.get("role") != "tool":
        return False
    return True

# ====L1====  切割中间的消息
def snip_compact(messages, max_messages=50):
    if len(messages) <= max_messages:
        return messages
    # 这两个参数做切割强制保留的开头和结尾段数
    keep_head, keep_tail = 3, max_messages - 3
    # 这两个参数做指针
    head_end, tail_start = keep_head, len(messages) - keep_tail
    # 判断第三条消息是否是assistant并且调用了工具
    if head_end > 0 and _message_has_tool_calls(messages[head_end - 1]):
        # 如果调用了工具就往后移一个 调用了两个工具就移两个移到工具调用结束的地方
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    #这里判断在tail_start位置有没有工具结果和这个工具结果前面没有调用工具的语句 如果有的话还要调整指针位置向前移动  但指针之间的间隔不用必须50
    if (tail_start > 0 and tail_start < len(messages)) and _is_tool_result_message(messages[tail_start]) and _message_has_tool_calls(messages[tail_start - 1]):
        tail_start -= 1
    # 设置边界兜底如果没有多余消息可删
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]

# ====L2====  拿到所有工具结果看返回的消息内容多吗如果多的话就会直接舍弃 放进去一个说明
# 该函数将工具结果收集起来  【结果blocks是一个列表里面是一个个元组由索引与tool的内容组成】
def collect_tool_results(messages):
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "tool" or not isinstance(msg.get("content"), str):
            continue
        msg_ture=msg["content"]
        blocks.append((mi,msg["content"]))
    return blocks

#压缩工具结果
def micro_compact(messages):
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # 拿到所有的工具结果  选择在KEEP_RECENT之前的工具结果
    for (idx, content) in tool_results[:-KEEP_RECENT]:
        if len(content) > 120:
            messages[idx]["content"] = "[之前的消息结果过于紧凑]"
    return messages

# ====L3==== 压缩工具结果放入文件中
#该函数创建目录将满足长消息写入文件
def persist_large_output(tool_use_id, output):
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

# 限制单论对话中所有tool的结果的文本总长度防止工具返回的内容过长塞满上下文 满足条件的话就把过长的话放进文件留一个地址给原来的content
def tool_result_budget(messages, max_bytes=200_000):
    blocks = []
    # 从对话末尾倒着遍历，只取本轮刚返回的tool，碰到其它role立刻停止 最终将blocks列表加入所有工具的字典
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            break
        # 把从后往前的tool放每回的第一个就能得到一个原有的顺序
        blocks.insert(0, msg)

    #将每一个tool分成索引和内容变成元组放进列表  msg就是一个个字典里面是role：tool，content：内容
    blocks = [(idx, msg) for idx, msg in enumerate(blocks)]
    #将里面所有tool的content内容加在一起判断长度
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    # 总长度没超限，直接返回原对话
    if total <= max_bytes:
        return messages

    # key会把这个装满元组的列表直接打开放入每个元组作为p    按长度排序
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)

    for _, block in ranked:
        # 总长度达标，停止压缩
        if total <= max_bytes:
            break
        content = str(block.get("content", ""))
        # 单条内容很短，跳过
        if len(content) <= PERSIST_THRESHOLD:
            continue
        # OpenAI字段是 tool_call_id
        tid = block.get("tool_call_id", "unknown")
        # 原地替换这条tool消息的content（不删除消息，只替换文本）
        block["content"] = persist_large_output(tid, content)
        # 重新计算本轮所有工具总字符长度
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)

    # 返回修改完成的对话列表
    return messages

# ====L4====  自动压缩
# 将所有对话消息写入jsonl文件  【返回的是所有对话的jsonl文件路径】
def write_transcript(messages):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    # 这里time.time()返回的是float 所以可以使用int()
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path

# 压缩对话ai并且作为工具被调用
def summarize_history(messages):
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    response = client.chat.completions.create(model="deepseek-v4-pro", messages=[{"role": "system", "content": prompt}], max_tokens=2000)
    return "\n".join(msg.message.content for msg in response.choices if msg.message.content != None and msg.message.tool_calls == None )


# <====== ai自动压缩上下文工具函数 ======>     集结上两个函数功能构成这个工具函数
def compact_history_tool(messages):
    transcript_path = write_transcript(messages)
    print(f"[所有对话已保存: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ====L5==== 紧急处理上下文 在api错误时
def emergency_compact(messages):
    transcript_path = write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    # 判断 这个tail_start这个指针距离终点前五个的位置  是否有tool和这个tool上一句话有没有tool_calls 最终达到没有调用工具的位置
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_calls(messages[tail_start - 1])):
        tail_start -= 1
    # 到达那里开始对上下文切片 然后放进总结ai总结函数 拿到summary  重新变成user消息
    summary = summarize_history(messages[:tail_start])
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]





# =========所有工具的JSON Schema==============================================================================================

# 主工具说明
TOOLS = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "运行一个命令去工作",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "设置一个命令"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "read",
            "description": "读取一个文件的内容",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "limit":{
                        "type":"integer",
                        "description":"限制读取的行数"
                    }
                }
            }
        }
    },{


        "type": "function",
        "function": {
            "name": "write",
            "description": "写入一个文件的内容",
            "parameters": {
                "type": "object",
                "required": ["path","content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content":{
                        "type":"string",
                        "description":"写入的内容"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "edit",
            "description": "编辑一个文件",
            "parameters": {
                "type": "object",
                "required": ["path","old_content","new_content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "old_content":{
                        "type":"string",
                        "description": "要被替换的内容"
                    },
                    "new_content":{
                        "type":"string",
                        "description": "新的内容"
                    }
                }
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "glob",
            "description": "检索文件",
            "parameters": {
                "type": "object",
                "required": ["patten"],
                "properties": {
                    "patten": {
                        "type": "string",
                        "description": "检索的通配符"
                    }
                }
            }
        }
    },
    {
        "type": "function",
         "function": {
             "name": "todo_write",
             "description": "更新全部待办任务清单",
             "parameters": {
                 "type": "object",
                 "required": ["todos"],
                 "properties": {
                     "todos": {
                         "type": "array",
                         "items": {
                             "type": "object",
                             "required": ["content", "status"],
                             "properties": {
                                 "content": {"type": "string", "description": "待办任务描述文本"},
                                 "status": {
                                     "type": "string",
                                     "enum": ["pending", "in_progress", "completed"],
                                     "description": "任务状态：待处理/进行中/已完成"
                                 }
                             }
                         }
                     }
                 }
             }
         }
     }
    ,{
        "type": "function",
        "function": {
         "name": "subagent",
         "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
         "parameters": {
             "type": "object",
             "properties": {
                 "description": {
                     "type": "string",
                     "description": "Detailed description of the subtask that subagent needs to complete"
                 }
             },
             "required": ["description"]
         }
     }
    }
,{
   "type": "function",
   "function": {
     "name": "load_skill",
     "description": "Load the full content of a skill by name.",
     "parameters": {
       "type": "object",
       "title": "load_skill input schema",
       "properties": {
         "name": {
           "type": "string",
           "description": "The unique name identifier of the target skill to load."
         }
       },
       "required": ["name"],
       "additionalProperties": False
     }
   }
},{
    "type": "function",
    "function": {
        "name": "compact",
        "description": "Summarize earlier conversation to free context space.",
        "parameters": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "The full conversation messages list to compact",
                    "items": {"type": "object"}
                }
            }
        }
    }
}
,{

    {
        "type": "function",
        "function": {
            "name": "run_create_task",
            "description": "创建新任务，可指定前置依赖任务，生成任务文件存入本地",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "任务简短标题/主题"
                    },
                    "description": {
                        "type": "string",
                        "description": "任务详细需求描述",
                        "default": ""
                    },
                    "blockedBy": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "前置依赖任务id列表，这些任务全部完成后当前任务才可执行，无依赖则不传",
                        "default": []
                    }
                },
                "required": ["subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_list_tasks",
            "description": "列出系统所有任务，展示任务id、状态、执行者、依赖信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_get_task",
            "description": "根据任务id获取单个任务完整详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "目标任务的id"
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_claim_task",
            "description": "申领待执行任务，校验依赖通过后标记任务为执行中、分配给agent执行",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "要申领的任务id"
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_complete_task",
            "description": "将正在执行的任务标记为已完成，自动解锁依赖该任务的下游任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "要完成的任务id"
                    }
                },
                "required": ["task_id"]
            }
        }
    }

},{
    {
  "type": "function",
  "function": {
    "name": "run_schedule_cron",
    "description": "创建cron定时任务，到指定时间自动触发对应提示词执行",
    "parameters": {
      "type": "object",
      "required": ["cron", "prompt"],
      "properties": {
        "cron": {
          "type": "string",
          "description": "标准5位cron表达式：分 时 日 月 星期，例如 0 9 * * 1 代表每周一早上9点"
        },
        "prompt": {
          "type": "string",
          "description": "定时触发后需要执行的指令内容"
        },
        "recurring": {
          "type": "boolean",
          "description": "是否重复执行；true周期性重复，false仅执行一次",
          "default": True
        },
        "durable": {
          "type": "boolean",
          "description": "是否持久化保存，程序重启后任务仍保留",
          "default": True
        }
      }
    }
  }
}
},{
    {
  "type": "function",
  "function": {
    "name": "run_list_crons",
    "description": "查询系统内所有已创建的定时任务，返回任务id、cron、执行内容、重复/持久化属性",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  }
}
},{
    {
  "type": "function",
  "function": {
    "name": "run_cancel_cron",
    "description": "根据任务id删除指定定时任务，同时同步删除持久化文件内的记录",
    "parameters": {
      "type": "object",
      "required": ["job_id"],
      "properties": {
        "job_id": {
          "type": "string",
          "description": "需要取消的定时任务唯一编号，可通过run_list_crons获取"
        }
      }
    }
  }
}
}]


TOOL_HANDERS = {
    "bash": run_bash,
    "read":run_read,
    "write":run_write,
    "edit":run_edit,
    "glob":run_glob,

    "todo_write":run_todo_write,

    "subagent":spawn_agent,

    "load_skill":load_skill,
    "create_task": run_create_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,

    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
}

#==========后台任务=====================================================================================================================
#后台任务编号自增器
_bg_counter = 0
#存放所有正在运行/已结束的后台任务的基础信息
background_tasks: dict[str, dict] = {}   # bg_id → {tool_use_id, command, status}
#存放后台任务跑完后的输出内容
background_results: dict[str, str] = {}   # bg_id → output
#多线程锁
background_lock = threading.Lock()


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """判断命令是否耗时"""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)

def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断命令是否需要后台运行  内含is_slow_operation判断是否是耗时操作"""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


def execute_tool(block) -> str:
    """执行工具 返回结果"""
    handler = TOOL_HANDERS.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


def start_background_task(block) -> str:
    """
    1.生成后台任务编号
    2.定义子线程任务worker
       子线程内部调用execute_tool跑完整工具逻辑;执行完成后,加锁修改共享字典:把任务状态改为completed
       保存执行输出到background_results
    3.主线程登记任务
       主线程先获取锁,往background_tasks中登记任务信息 完成登记后释放锁
    4.返回任务编号

    这里就是将命令开始运行然后将其挂载后台执行 并存储下信息就让后台自己运行 函数先执行退出了
    """
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"[background] dispatched {bg_id}: {cmd[:40]}")
    return bg_id



def collect_background_results() -> list[str]:
    """收集已经执行完成的后台任务,整理成通知文本,交给大模型知晓后台任务跑完了"""
    #筛选出已完成任务  加锁防止子线程正在修改任务状态时读取到的错乱数据:上锁原理就是线程a执行到这里会把锁站住此时其他线程执行执行同一行时,会卡在这里,等待锁被释放
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        #把任务执行结果从全局字典删除 以便不反复项模型推送同一条任务完成通知 pop修改了共享字典所以再次加锁保护
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"[background done] {bg_id}: "
              f"{task['command'][:40]} ({len(output)} chars)")
    return notifications


#===========动态构成system prompt======================================================================================
"""
updata_context就是负责看消息列表判断需不需要加上memoies的参数也就是MEMORY里的内容 返回context 
然后用get_system_prompt来看名不命中缓存如果命中还用上一次的prompt也就是_last_prompt若没有命中的话嗲
用assemble_system_prompt重新组装prompt变成_last_prompt被用作系统提示词
"""

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}

# 组装系统prompt    【看是否有memories内容有的话就放进去】
def assemble_system_prompt(context: dict) -> str:
    sections = []

    # 将预备好的内容先放入列表中
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["tools"])
    sections.append(PROMPT_SECTIONS["workspace"])

    # 如果context有memory就放入列表中
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)

_last_context_key = None
_last_prompt = None

# 判断updata_context拿到context  【跟据拿到的context是否命中缓存来获取新的还是老的prompt】
def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt
    
    # 将context转换成json字符串作为缓存的key
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("[cache hit] system prompt unchanged")
        return _last_prompt
    
    # 没命中的话就将context等于key 以便于下次命中
    _last_context_key = key
    # 若命中就根据这次context拼接出prompt
    _last_prompt = assemble_system_prompt(context)

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"[assembled] sections: {', '.join(loaded)}")
    return _last_prompt

# 返回的就是context  【每次调用直接拿到的是MEMORY的内容也就是所有记忆文件的结合】
def update_context(context: dict) -> dict:
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": list(TOOL_HANDERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }



#========防止模型越级操作———放入钩子函数=================================================================================================
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

# gata1：否定命名列表
def check_deny_list(command:str):
    for deny in DENY_LIST:
        if deny in command:
            return "error:出现了禁用命令"
    return None

# gata2: 规则匹配
PERMISSION_ROLES=[{
    "tools":["write","edit"],
    "check":lambda args: not (WORKDIR/args.get("path","")).resolve().is_relative_to(WORKDIR),
    "messages":"跳出工作目录的写入或编辑操作"
},{
    "tools":["bash"],
    "check":lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
    "messages":"危险的bash命令"
}]
#两个lambda函数都是返回一个布尔值

#它是拿ai返回的消息结果进行匹配
def check_rules(tool_name:str,args:dict):
    for role in PERMISSION_ROLES:
        if tool_name in role["tools"] and role["check"](args):
            return role["messages"]
    return None

# gata3: 用户权限控制
def ask_user() -> str:
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# 三个gata合三为一    这个如同gata2一样，根据模型的结果放进这个函数进行一系列比对拿答案的
"""
def check_permission(block) -> bool:
    if block.name == "bash":
        args = json.loads(block.arguments)
        result=check_deny_list(args.get("command",""))
        if result:
            return False
    reason=check_rules(block.name,json.loads(block.arguments))
    if reason:
        decision = ask_user()
        if decision == "deny":
            return False
    return True
"""

#========钩子函数=========================================================================================================
DESTRUCTIVE=["rm ", "> /etc/", "chmod 777"]



HOOKS={
    "BEFORE_AGENT":[],
    "BEFORE_TOOL":[],
    "AFTER_TOOL":[],
    "AFTER_AGENT":[]
}

def register_hook(event:str,callback):
    HOOKS[event].append(callback)


def trigger_hook(event:str,*args):
    for callback in HOOKS[event]:
        result=callback(*args)
        if result:
            return result
    return None

# <====五种钩子函数只有permission_check函数是没有触发打印的条件的但是它是由log_function钩子函数触发====>

# # 防越级操作钩子函数（三合一)  替代升级了上面的三合一的越级操作但功能一致 只是要将其注册为了钩子函数
def permission_check_hook(block):
    if block.function.name == "bash":
        args = json.loads(block.function.arguments)
        for pattern in DENY_LIST:
            if pattern in args.get("command", ""):
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in args.get("command", ""):
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.function.name in ("write","edit"):
        args = json.loads(block.function.arguments)
        path = args.get("path","")
        if not (WORKDIR/path).resolve().is_relative_to(WORKDIR):
            ask_result = ask_user()
            if ask_result=="deny":
                return "请求被用户拒绝"
    return None

# 日志钩子函数 记录每一个工具的调用信息
def log_function_hook(block):
    args=json.loads(block.function.arguments)
    args_preview =str(list(args.values())[:2])[:50]
    print(f"[HOOK] {block.function.name} {args_preview}")
    return None

#最大输出警告钩子函数
def large_output_hook(block,output):
    if len(str(output)) > 10000:
        print(f"[HOOK]:输出过长，请查看日志")
    return None



# 看上下文工作地址的工具函数
def context_inject_hook(query:str):
    print(f"[HOOK] 检查工作地址{WORKDIR}")
    return None


# 看工具调用数量的钩子函数
def summary_hook(messages:list):
    tool_count = sum(1 for m in messages if m.get("role","") =="tool")
    print(f"[HOOK] 工具调用数量{tool_count}")
    return None

register_hook("BEFORE_AGENT",context_inject_hook)
register_hook("BEFORE_TOOL",permission_check_hook)
register_hook("BEFORE_TOOL",log_function_hook)
register_hook("AFTER_TOOL",large_output_hook)
register_hook("AFTER_AGENT",summary_hook)

#=========agent循环========================================================================================================  =================

count_todo=0

# 看是否超过这个数值 超过了就无法在紧急压缩上下文了
MAX_REACTIVE_RETRIES = 1
def agent_loop(messages:list,context):

    #这个参数是用来计数 达到某个数值就确定自己目标防止跑题
    global count_todo

    #这个参数用来判断是否触发紧急压缩上下文
    reactive_retries = 0

    #<===6===>
    #加载出所有文件中相关记忆的内容合并成的字符串
    memories_content = load_memories(messages)
    #len-1是为了能够下标取值刚好取到最新的消息
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None


    # <===7===>
    # 加入系统提示词  根据context  每轮agent_loop循环一次，重新生成一次看是否命中缓存
    system = get_system_prompt(context)
    if not messages or messages[0].get("role") != "system" or messages[0]["content"] != system:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system
        else:
            messages.insert(0, {"role": "system", "content": system})
    
    #<===8===>
    #定义error_recovery的状态
    state = RecoveryState()

    max_tokens = DEFAULT_MAX_TOKENS


    while True:

        # 消费 cron 队列，将已触发的定时任务注入为 user 消息
        fired = consume_cron_queue()
        for job in fired:
            messages.append({
                "role": "user",
                "content": f"[Scheduled] {job.prompt}"
            })
            print(f"  [inject cron] {job.prompt[:50]}")
        
        #<===6===>
        #拿到干净的对话消息不要tool_calls的
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role",""),
            "content": str(m.get("content",""))} for m in messages]

        #<====5====>    
        messages[:] = tool_result_budget(messages)    # L3: persist large results first
        messages[:] = snip_compact(messages)          # L1: trim middle
        messages[:] = micro_compact(messages)         # L2: old result placeholders

        #该段就是截断循环拿到之前所有的对话如果超过了法制会直接调用ai压缩重新变成一条总结user消息
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history_tool(messages)
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system})

        #<====4====>
        if count_todo>3 and messages:
            messages.append({"role": "user","content": f"注意请更新你的todos"})
            count_todo=0

        #<====6====>
        # 把长期记忆拼进用户对话中  如果满足触发条件
        request_messages = messages
        if memories_content and memory_turn is not None and memory_turn < len(messages):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }

        try:

            #<====1====><====8====PATH3>
            #429 和 529 统一走指数退避 + 抖动：第一次等 0.5 秒，第二次等 1 秒，第三次等 2 秒，最多 10 次。加随机抖动让并发请求不在同一时刻重试。连续 3 次 529 过载 → 切换到备用模型
            response = with_retry(lambda mt=max_tokens ,mdl=state.CURRENT_MODEL:client.chat.completions.create(
            model=mdl,
            messages=request_messages,
            tools=TOOLS,
            max_tokens=max_tokens,
            ),state)
            reactive_retries = 0  # reset on successful API call
        
        except Exception as e:
            
            # <====8====PATH2>
            # 四层压缩都无法进行的话 就触发这里的解决
            # error recovery  如果经过紧急压缩还是不行的话就会触发error recovery
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    messages[:] = error_recovey_energency_compact(messages)
                    state.has_attempted_reactive_compact = True
                    continue
                print("[unrecoverable] still too long after compact")
                messages.append({"role":"assistant","content":"[Error] Context too large, cannot continue."})
                return

            #错误无法covery
            name = type(e).__name__
            print(f"[unrecoverable] {name}: {str(e)[:100]}")
            messages.append({"role": "assistant", "content":f"[Error] {name}: {str(e)[:200]}"})
            return

            # # <====5====>
            # # 这里是触发紧急压缩 之前的消息只会保留最后五个和加上总结的消息
            # if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
            #     print("[reactive compact]")
            #     # *紧急压缩但他会把system也压缩掉所以在这里补上
            #     messages[:] = emergency_compact(messages)
            #     if not messages or messages[0].get("role") != "system":
            #         messages.insert(0, {"role": "system", "content": system})
            #     reactive_retries += 1
            #     continue
            # raise


        #<====2====><====8====PATH1>
        #第一次发生时，直接把 max_tokens 从 8K 升级到64K(8倍空间),重试同一请求——此时不追加截断输出到 messages，保持原始请求不变。如果 64K 还是不够，才保存截断输出并注入续写提示让模型接着刚才的话继续说，最多 3 次
        #如果模型返回的max_tokens触发了就不会再走下面的将response消息添加到messages中 所以在这里就要添加了
        if response.choices[0].finish_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f" [max_tokens] escalating"
                      f" {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}")
                continue
            # 64K still truncated: save truncated output + continuation prompt
            messages.append({"role": "assistant", "content": response.choices[0].messages.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f" [max_tokens] continuation"
                      f" {state.recovery_count}/{MAX_RECOVERY_RETRIES}")
                continue
            print(" [max_tokens] recovery limit reached")
            return



        # <====1=====>
        #这里就是处理模型回复的消息将内容放入到messages中若有tool_calls也放入
        assistant_msg = response.choices[0].message
        msg = {"role": "assistant", "content": assistant_msg.content}
        # 这里就是添加content和tool_calls 判断有的话就添加  先正常放入content
        if assistant_msg.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in assistant_msg.tool_calls]
        messages.append(msg)


        # <====2=====>
        # "分叉路口"如果没有工具调用就在此离开不然就继续进行工具的调用
        if response.choices[0].finish_reason != "tool_calls":
            # ==钩子函数==
            trigger_hook("AFTER_AGENT", messages)
            
            # <====6=====>
            # ==触发记忆函数==
            # 第一个是创建新的多个memory文件 第二个整合删去老的重复的创建新的
            # 上面定义的时候只能拿到到提问的时候内容在这里把模型回复的消息加上
            pre_compress.append(response.choices[0].message.content)
            extract_memories(pre_compress)
            consolidate_memories()
            return





        # 从此相当于进入到了下一轮因为它在上一步并没有离开
        count_todo+=1

        
        #background工具结果
        results_background = []
        # <====3====>
        # arg_special是钩子函数的block args是工具函数的block参数
        for tc in response.choices[0].message.tool_calls:

            tc_ture= json.loads(tc.function.arguments)
            # <====5=====><====3====>虽然介绍了函数但没有在tool_handler中定义
            if tc.function.name == "compact":
                messages[:] = compact_history_tool(messages)
                messages.append({"role": "user", "content": "Compacted. Conversation history has been summarized."})
                # *紧急压缩但他会把system也压缩掉所以在这里补上
                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": system})
                break  # end current turn, start fresh with compacted context

           
            # 函数调用前钩子函数放在这里 可以确保后台执行或直接运行函数都可以运行
            arg_special=tc
            # ====钩子函数==== 
            blocked=trigger_hook("BEFORE_TOOL", arg_special)
            if blocked:
                #虽然接受钩子函数的返回值并写入tool里面但是这些钩子函数都没有返回值
                messages.append({"role":"tool","content":blocked})
                continue

            #<====9====> 
            #这样确保函数能够能够正常执行参数 因为openai返回的参数是对象
            class ToolCallAdapter:
                def __init__(self, tc):
                    self.id = tc.id
                    self.name = tc.function.name
                    self.input = json.loads(tc.function.arguments)

            block = ToolCallAdapter(tc)
            # 模型触发了几次后台任务就添加几个results_background里 里面会有多个tool但openai就是可以连着有多个   
            # 这里只是提取名字进入列表
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results_background.append({"role": "tool", "tool_call_id": tc.id,
                                "content": f"[Background task {bg_id} started] ..."})
                # 这个continue很关键  <因为先判断是否工具函数要后台执行如果需要的continue 就会跳到下一个循环 没有的话就按照正常执行工具函数>
                continue
            


            #==进行工具调用handler==  调用工具就是把模型想要工具参数放进函数里，里面都是自动化开始处理
            handler = TOOL_HANDERS.get(tc.function.name)
            arg_special=tc
            args = json.loads(tc.function.arguments)  #这么写是因为ai返回的response的json字符串必须先转化成python字典才能解包或者用[]
            if handler:  # handler的参数可以由args随意提供因为我们使用的参数都是大模型提供，我只需要解包
                output = handler(**args) if isinstance(args, dict) else handler(args)#单工具的调用output = run_bash(args["command"])
            else:
                output = f"error: unknown tool {tc.function.name}"
            # ====钩子函数====
            trigger_hook("AFTER_TOOL", arg_special, output)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


        # <====9====>
        # 1. 把后台任务的占位 tool 结果逐条写入 messages
        for r in results_background:
            messages.append(r)

        # 2. 收集已完成的后台任务，作为 user 消息通知模型
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                messages.append({"role": "user", "content": notif})
            print(f"  [inject] {len(bg_notifications)} background notification(s)")







# ── 会话级全局变量（供 run_agent_turn_locked / queue_processor_loop 使用）──
session_history: list = []
session_context: dict = {}


def run_agent_turn_locked(user_query: str | None = None):
    """
    加锁执行一轮 agent 回合。
    手动输入和 cron 触发都走这里，agent_lock 保证互斥。
    """
    
    global session_context
    if user_query is not None:
        session_history.append({"role": "user", "content": user_query})
    agent_loop(session_history, session_context)
    session_context = update_context(session_context)
    
    # 打印最新 assistant 文本
    if session_history:
        last = session_history[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            content = last.get("content", "")
            if isinstance(content, str):
                print(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(block.get("text", ""))
    print()


def thread_queue_processor_loop():
    """
    后台线程：当 cron 触发了任务且 agent 空闲时，自动跑一轮 agent。

    工作机制：
    1. 每 0.2 秒检查 cron_queue 是否有待处理任务
    2. 如果有，尝试拿 agent_lock（非阻塞）
       - 拿到了 → agent 空闲 → 自动推送 cron 消息给模型
       - 没拿到 → agent 正在处理用户请求 → 下轮再试
    """
    while True:
        time.sleep(0.2)
        if not has_cron_queue():
            continue
        if not agent_lock.acquire(blocking=False):
            continue
        try:

            if not has_cron_queue():          # 双重检查
                continue
            print("\n  [queue processor] delivering scheduled work")
            
            # ----自动触发进入agent----
            run_agent_turn_locked()
        finally:
            agent_lock.release()


if __name__ == "__main__":
    # 初始化 session 级别变量
    session_context = update_context({})

    # 启动 cron 自动推送线程
    threading.Thread(target=thread_queue_processor_loop, daemon=True).start()
    print("  [queue processor] started")

    print("输入消息（exit/quit 退出）：")
    while True:
        try:
            query = input(">> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit", "quit", ""):
            break
        
        # ----手动输入进入agent----
        # 用 agent_lock 包裹，和 cron 自动触发互斥
        with agent_lock:
            run_agent_turn_locked(query)




