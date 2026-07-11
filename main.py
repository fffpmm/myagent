import json
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path



load_dotenv(override=True)
WORKDIR = Path.cwd()

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")





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
    }]


TOOL_HANDERS = {
    "bash": run_bash,
    "read":run_read,
    "write":run_write,
    "edit":run_edit,
    "glob":run_glob
}

#=========钩子函数==================
HOOKS={
    "BEFORE_AGENT":[],
    "BEFORE_TOOL":[],
    "AFTER_TOOL":[],
    "AFTER_AGENT":[]
}

def register_hook(event:str,callback):
    HOOKS[event].append(callback)


def trigger_hook(event:str,*args):
    for callback in HOOKS(event):
        result=callback(*args)
        if result:
            return result
    return None





#========防止模型越级操作===========
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

def check_rules(tool_name:str,args:dict):
    for role in PERMISSION_ROLES:
        if tool_name in role["tools"] and role["check"](args):
            return role["messages"]
    return None

# gata3: 用户权限控制
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# 三个gata合三为一   
def check_permission(block) -> bool:
    if block.name == "bash":
        args = json.loads(block.arguments)
        result=check_deny_list(args.get("command",""))
        if result:
            return False
    reason=check_rules(block.name,json.loads(block.arguments))
    if reason:
        decision = ask_user(block.name, block.arguments, reason)
        if decision == "deny":
            return False
    return True




#========防止模型越级操作===========
def agent_loop(messages:list):
    while True:
        response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=TOOLS,
        max_tokens=8000
        )
    
        # messages.append({"role":"assistant","content":response.choices[0].message.content}) 不能这样是因为openai和Claude的contennt不一样openai的message里有content和tools_calls两者是分开的而claude的content直接两者都是一个列表的
        assistant_msg = response.choices[0].message

        msg = {"role": "assistant", "content": assistant_msg.content}
        if assistant_msg.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in assistant_msg.tool_calls]
        messages.append(msg)
        
        if response.choices[0].finish_reason != "tool_calls":
            return
            

        result =[]
        for tc in response.choices[0].message.tool_calls: #这里之所以要用for循环是因为可能会有多个工具调用
            #==1.先进行权限判断check_permission==
            if not check_permission(tc.function):  # 传函数名(字符串)，不是function对象
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "拒绝访问"})
                continue
            
            #==2.进行工具调用handler==
            handler = TOOL_HANDERS.get(tc.function.name)
            args = json.loads(tc.function.arguments)#这么写是因为ai返回的response的json字符串必须先转化成python字典才能解包或者用[]
            if handler:
                output = handler(**args) if isinstance(args, dict) else handler(args)#单工具的调用output = run_bash(args["command"])
            else:
                output = f"error: unknown tool {tc.function.name}"
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

            agent_loop(history)
            print(history)
            