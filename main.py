import ast
import json
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

# SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

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

"""   该agent在运行功能时会产生的系统目录与文件
.task_outputs / tool-results/<tool_use_id>.txt
.transcripts / transcript_<timestamp>.jsonl
skills/<skill_name>和SKILL.md
"""

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")


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
        if not d.is_file():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
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
def build_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )


#只给主agent进行调用 subagnet就不给skill了
SYSTEM = build_system()

#<====工具函数====>  给主agent进行调用 真正让模型调用的工具函数 想要使用时传来的技能名就能拿到内容
def load_skill(name: str) -> str:
    """Load full skill content. Lookup via registry — no path traversal."""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


#============工具函数==================================================================================================
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



#========模型调用工具函数（该工具不进行外部操作只做提醒和打印在终端）======================================================================

# 对todos的消息类型进行检测 只要列表里是字典的   todos这个参数也是像其他工具函数一样的由大模型定义给它
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

#<====工具函数====>  为状态添加上符号并打印出每个任务的状态  todos这个参数也是像其他工具函数一样的由大模型定义给它
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

# block传来的将是messages也就是消息列表
def extract_text(block):
    return "\n".join(b.content for b in block if not b.tool_calls and b.content)

#<====工具函数====>
def spawn_agent(description):
    print("[SPAWN AGENT]")
    messages =[{"role":"user","content":description}]

    for _ in range(30):
        response = client.chat.completions.create(
            model = "deepseek-v4-pro",
            messages = messages,
            system =SUB_SYSTEM,
            tools = SUB_TOOLS,
            max_tokens=8000
        )
        meg = {"role":"assistant","content":response.choices[0].message.content}
        if response.choices[0].message.tool_calls:
            # 主agent用的是for循环一个个拿字典的key和value 这里直接model_dump
            messages.tool_calls = response.choices[0].message.tool_calls.model_dump()
        messages.append(meg)

        if response.choices[0].finish_reason != "tool_calls":
            trigger_hook("AFTER_AGENY",messages)
            break


        for tc in response.choices[0].message.tool_calls:
            if not tc:
                continue
            blocked=trigger_hook("BEFORE_TOOL", tc)
            arg=json.loads(tc.args)
            output=SUB_TOOL_HANDERS[tc.name](**arg)
            print(f"[sub] {tc.name}: {str(output)[:100]}")
            messages.append({"role":"tool","content":output})
    
    result=extract_text(messages)
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
# =========四层上下文消息压缩===============================================================================================
#前三层如hook放入agent_loop里自动判断触发第四层是工具函数由模型调用

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
    # 从对话末尾倒着遍历，只取本轮刚返回的tool，碰到其他role立刻停止 最终将blocks列表加入所有工具的字典
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


# <====ai自动压缩上下文工具函数====>     集结上两个函数功能构成这个工具函数
def compact_history(messages):
    transcript_path = write_transcript(messages)
    print(f"[所有对话已保存: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ====L5==== 紧急处理上下文 在api错误时
def reactive_compact(messages):
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
                "focus": {
                    "type": "string"
                }
            }
        }
    }
}
]


TOOL_HANDERS = {
    "bash": run_bash,
    "read":run_read,
    "write":run_write,
    "edit":run_edit,
    "glob":run_glob,
    "todo_write":run_todo_write,
    "subagent":spawn_agent,
    "load_skill":load_skill,
    "compact":reactive_compact
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
def agent_loop(messages:list):
    global count_todo
    while True: 
        #<====5====>
        messages[:] = tool_result_budget(messages)    # L3: persist large results first
        messages[:] = snip_compact(messages)          # L1: trim middle
        messages[:] = micro_compact(messages)         # L2: old result placeholders

        #该段就是截断循环拿到之前所有的对话如果超过了法制会直接调用ai压缩重新变成一条总结user消息
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        #<====4====>
        if count_todo>3 and messages:
            messages.append({"role": "user","content": f"注意请更新你的todos"})
            count_todo=0

        response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=TOOLS,
        max_tokens=8000
        )
    
        # <====1=====>
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
        # “分叉路口”如果没有工具调用就在此离开不然就继续进行工具的调用
        if response.choices[0].finish_reason != "tool_calls":
            # ==钩子函数==
            trigger_hook("AFTER_AGENT", messages)
            return

        # 从此相当于进入到了下一轮因为它在上一步并没有离开
        count_todo+=1


        # <====3====>
        # arg_special是钩子函数的block args是工具函数的block参数
        for tc in response.choices[0].message.tool_calls:
            if not tc:
                continue
            arg_special=tc
            # ==钩子函数== 虽然接受钩子函数的返回值并写入tool里面但是这些钩子函数都没有返回值
            blocked=trigger_hook("BEFORE_TOOL", arg_special)
            if blocked:
                messages.append({"role":"tool","content":blocked})
                continue
        
    
        
            #==进行工具调用handler==  调用工具就是把模型想要工具参数放进函数里，里面都是自动化开始处理
            handler = TOOL_HANDERS.get(tc.function.name)
            arg_special=tc
            args = json.loads(tc.function.arguments)  #这么写是因为ai返回的response的json字符串必须先转化成python字典才能解包或者用[]
            if handler:  # handler的参数可以由args随意提供因为我们使用的参数都是大模型提供，我只需要解包
                output = handler(**args) if isinstance(args, dict) else handler(args)#单工具的调用output = run_bash(args["command"])
            else:
                output = f"error: unknown tool {tc.function.name}"
            # ==钩子函数==
            trigger_hook("AFTER_TOOL", arg_special, output)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})   

        

    
 


if __name__ == "__main__":
    #history进agent_loop相当于变成messages了所以append都会进入到history
    history = []
    history.append({"role":"system","content":SYSTEM})
    while True:
            try:
                query=input("请输入消息(exit和quit是退出程序)：")
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower().strip() in ["exit" ,"quit",""]:
                break
            history.append({"role":"user","content":query})

            # ==钩子函数==
            trigger_hook("BEFORE_AGENT", query)
            agent_loop(history)

            print(history)
            