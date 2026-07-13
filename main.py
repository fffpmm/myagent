import ast
import json
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path



load_dotenv(override=True)
#给s3用
WORKDIR = Path.cwd()
#给s4用
CURRENT_TODOS:list[dict]=[]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)




client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")




#============工具函数=================================================================================================
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

#为状态添加上符号并打印出每个任务的状态  todos这个参数也是像其他工具函数一样的由大模型定义给它
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


def spawn_agent(description):
    print("[SPAWN AGENT]")
    messages =[{"role":"user","content":description}]

    for _ in range(30):
        response = client.completions.create(
            model = "deepseek-v4-pro",  
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
]


TOOL_HANDERS = {
    "bash": run_bash,
    "read":run_read,
    "write":run_write,
    "edit":run_edit,
    "glob":run_glob,
    "todo_write":run_todo_write,
    "subagent":spawn_agent
}


#========防止模型越级操作=================================================================================================
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
            