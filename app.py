"""
Virtual Data Analyst API
------------------------
Single-file FastAPI app that implements the workflow you asked for:

1. Accept a question.txt upload (or raw text).
2. Ask an LLM (via a proxy URL) to generate Python code that *loads/collects* the data
   into a pandas DataFrame named `df`. The model is instructed to use only pandas,
   duckdb and safe stdlib helpers provided.
3. Run the generated "data-fetch" code in a restricted execution environment and
   extract `df`.
4. Send a sample of the dataframe plus the original questions back to the LLM and
   ask it to generate Python code that computes the answers (again running in the
   restricted environment). The answer code should produce a Python variable
   `answers` (string or dict serializable to JSON).
5. Execute that answer-code and return the answers JSON string in the response.

SECURITY: This is *not* a production sandbox. It performs basic AST checks and
limits available modules, but running arbitrary model-generated code is inherently
risky. Run inside an isolated environment (container, restricted VM) and/or with
further hardened execution before exposing to untrusted users.

Usage:
    pip install fastapi uvicorn pandas duckdb requests python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /analyze  - multipart upload: file: question.txt  OR raw text field 'text'

"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import io
import json
import ast
import textwrap
import uuid
import requests
import pandas as pd
import numpy as np
import sklearn
import duckdb
import multiprocessing
import traceback
import aiohttp
import base64
from typing import Optional, List

# ---------- Configuration ----------
LLM_PROXY_URL = "https://aipipe.org/openrouter/v1/chat/completions"
API_PROXY_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6InNtcml0aS50dXJhbkBnbWFpbC5jb20ifQ.9pqId92evbHHNgH8I5FWqTHYqdMSmo-bM2TnLi2XEgg"
# Set MODEL and any other settings your proxy accepts
LLM_MODEL = "gpt-4o-mini"  # change if needed
LLM_MAX_TOKENS = 1500

# Allowed modules and names for execution environment
ALLOWED_MODULES = {
    'pd': pd,
    'pandas': pd,
    'duckdb': duckdb,
    'io': io,
    'json': json,
}

# Simple blacklist for obviously dangerous operations (helps quick-reject)
BLACKLIST_PATTERNS = [
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "subprocess",
    "socket",
    "requests",
    "os.system",
    "sys.",
    "shutil",
    "ctypes",
    "Popen",
]

# ---------- FastAPI app ----------
app = FastAPI(title="Virtual Data Analyst API")


class AnalyzeResponse(BaseModel):
    success: bool
    answers: Optional[dict]
    error: Optional[str] = None


# ---------- Helpers: LLM calls ----------
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0):
    """Send a prompt to the proxy LLM endpoint and return the assistant content.

    This function assumes your proxy mirrors the OpenAI chat completions spec.
    Adapt the payload/headers if your proxy needs different fields.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
    }
    print('payload successfully updated')
    try:
        headers = {"Authorization": f"Bearer {API_PROXY_KEY}"}
        resp = requests.post(LLM_PROXY_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        print('data collected')
        # try a few likely locations for the returned text depending on proxy
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        if 'output' in data:
            return data['output'][0]
        # fallback: raw text
        return str(data)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


# ---------- Safety: AST checks ----------

def basic_safety_check(code: str) -> None:
    lowered = code.lower()
    for pat in BLACKLIST_PATTERNS:
        if pat in lowered:
            raise ValueError(f"Unsafe code pattern detected: {pat}")
    print('reached 1')

    tree = ast.parse(code)
    print('reached 2')
    for node in ast.walk(tree):
        # Detect exec() calls (Python 3 style)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'exec':
                raise ValueError("exec() is not allowed")
        # Reject import nodes that are not in allowed list
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name not in ('pandas', 'duckdb', 'io', 'json', 'matplotlib', 'seaborn', 'base64', 'numpy', 'sklearn'):
                    raise ValueError(f"Import of module '{name}' is not allowed")
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or '').split('.')[0]
            if mod not in ('pandas', 'duckdb', 'io', 'json', 'matplotlib', 'seaborn', 'base64', 'numpy', 'sklearn'):
                raise ValueError(f"Import from '{mod}' is not allowed")
        # Disallow dangerous attribute access
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in ('__dict__', '__class__', '__mro__', '__globals__'):
                raise ValueError("Access to dunder attributes is not allowed")
    print('reached final')


# ---------- Execution environment ----------
# def extract_and_run_code(code_text: str) -> str:
#     """Strip code fences, execute, and return output variable."""
#     # Remove triple backticks and 'python'
#     clean_code = re.sub(r"^```python\s*|```$", "", code_text.strip(), flags=re.MULTILINE)

#     local_vars = {}
#     try:
#         exec(clean_code, {"pd": pd, "duckdb": duckdb, "json": json}, local_vars)
#     except Exception as e:
#         raise RuntimeError(f"Generated code execution failed: {e}")

#     if "output" not in local_vars:
#         raise RuntimeError("No 'output' variable found in generated code")

#     return local_vars["output"]

def run_code(code_text, exec_globals, exec_locals, result_queue):
    try:
        compiled = compile(code_text, "<string>", "exec")
        exec(compiled, exec_globals, exec_locals)
        result_queue.put(exec_locals)
    except Exception as e:
        result_queue.put({"error": traceback.format_exc()})

def execute_with_timeout(code_text, exec_globals, exec_locals, timeout=150):
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    print('process started')
    process = multiprocessing.Process(
        target=run_code, 
        args=(code_text, exec_globals, exec_locals, result_queue)
    )
    process.start()
    process.join(timeout)
    print('process ended')
    if process.is_alive():
        process.terminate()
        process.join(2) 
        print('after 2s') # give it 2 seconds to clean up
        if process.is_alive():
            print('trying to kill') 
            process.kill() 
            print('killed') # force kill if still running
        raise RuntimeError("took too long")

    if not result_queue.empty():
        return result_queue.get()
    else:
        return {"error": "no output"}

        
# ---------- Code execution with timeout ----------


def execute_user_code(code: str, provided_env: dict = None, timeout: int = 150) -> dict:
    """Execute model-generated code with a tightly controlled globals dict.

    - `provided_env` are additional allowed names (like helpers)
    - Returns the namespace dict after execution.
    """
    print('safety check procesing...')
    basic_safety_check(code)
    print('safety check passed')
    exec_globals = {k: v for k, v in ALLOWED_MODULES.items()}
    print('exec globals done')
    # minimal builtins: expose only what's necessary
    safe_builtins = {
        'True': True,
        'False': False,
        'None': None,
        'len': len,
        'min': min,
        'max': max,
        'range': range,
        'print': print,
        'str': str,
        'int': int,
        'float': float,
        'dict': dict,
        'list': list,
        'tuple': tuple,
        'enumerate': enumerate,
        'sorted': sorted,
        "__import__": __import__, 
    }

    exec_globals['__builtins__'] = safe_builtins

    if provided_env:
        exec_globals.update(provided_env)

    print('globals executed')

    exec_locals = {}


    # execute
    try:
        compiled = compile(code, '<user-code>', 'exec')
        exec(compiled, exec_globals, exec_locals)
    except Exception as e:
        raise RuntimeError(f"Execution error: {e}")

    

    # Merge exec_globals + exec_locals for convenience
    result_ns = dict(exec_globals)
    result_ns.update(exec_locals)
    return result_ns

def _exec_retry_worker(code, queue):
    try:
        result = execute_user_code(code)
        queue.put({"success": True, "result": result})
    except Exception:
        queue.put({"success": False, "error": traceback.format_exc()})


async def get_image_description(image_bytes: bytes) -> str:
    # (Fill in your API endpoint, key, and request details here)
    api_url = "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyC4iKaY-zwLslbeFJ8owyVHfz7eawZgXII"

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    print(image_base64)

    request_body = {
        "requests": [{
            "image": {"content": image_base64},
            "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
        }]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=request_body) as resp:
            resp_json = await resp.json()
            print(resp.text)
            # Extract a useful textual description from response, e.g. labels
            labels = resp_json.get('responses', [{}])[0].get('labelAnnotations', [])
            description = ", ".join([label['description'] for label in labels])
            return description


# ---------- Prompt templates ----------

data_fetch_prompt_template = textwrap.dedent("""
You are a helpful assistant that writes Python code. The user will provide a small
"question.txt" which contains:
- one or more questions that need data
- context describing what data sources (URLs, CSV/Parquet files, SQL endpoints) may
  be needed

Write a Python script that loads/collects the required data into a single pandas
DataFrame named `df` (a pandas.DataFrame). The environment that will run the code
already has the following objects available and you should use them instead of
importing new modules: `pd` (pandas), `duckdb` (duckdb), `io`, `json`.

Guidelines (strict):
- Do NOT use `open`, `exec`, `eval`, or any subprocess or os.system calls.
- Only use the provided modules (pd, duckdb, io, json, bs$) and pure-Python stdlib.)` 
- The produced DataFrame must be assigned to a variable named `df`.
- you can use duckdb.connect() and con.execute(...) wherever necessary.
- Keep the code concise and only produce the Python code block (no explanation).
- After loading, ensure df is a pandas.DataFrame. If multiple files are loaded,
  concatenate them into one df and reset the index.
- No additional code to solve the questions should be there in the python code. On running the code, we should only get a dataframe.
- Do NOT include code for data processing, querries and graph plotting or any other process that not related to getting the data as dataframe

Respond with ONLY the Python code.
""")

answer_generation_prompt_template = textwrap.dedent("""
You are a helpful assistant that writes Python code to compute answers from a
pandas DataFrame named `df`. The runtime will provide `pd` and `duckdb` and has
already loaded a dataframe `df` (containing a sample of the user's data or the
entire dataset). The original user questions and any additional context are below.

Write Python code that computes a variable called `answers` (either a dict or a
string serializable to JSON) that contains the answers to the questions.
Guidelines:
- Use only pandas and duckdb for data processing.
- you can use sklearn for data processing if needed.
- STRICTLY Use pandas string replacement with a regex (.str.replace(r'[^\d.]', '', regex=True), that strips everything except digits and decimal points, and then convert to numeric to avoid errors.
- The code should not print; it should set `answers` and then finish.
- If numerical outputs are returned, ensure types are simple (int, float, str).
- Respond with ONLY the Python code.

Questions / Context:
""" + "{question_text}" + "\n\n" + "Data sample info:\n" + "{sample_info}" + "\n" + """

Respond with ONLY the Python code.
""")

retrydf_prompt = """
    You are a data extraction assistant that provides data based on context and questions.
    The user will give you a question and context. Write python code providing valid data from the souce specified.  
    You must return ONLY valid Python code that:
    1. Imports pandas
    2. Creates a pandas DataFrame named df
    3. Contains only the minimal data required to answer the question
    Do NOT print anything. Do NOT include explanations. 
    Do not output markdown formatting.
    """


# ---------- API Endpoint ----------

@app.post('/api', response_model=AnalyzeResponse)
async def analyze(request: Request, text: str = Form(None)):
    attachments = {}
    text_files = {}
    question_text = None
    image_descriptions = {}

    form = await request.form()
    print(form)
    for form_key, form_value in form.multi_items():
        print('file present')
        print('form key',form_value)
        print('instance', isinstance(form_value, UploadFile))
        if hasattr(form_value, "filename"):
            print('files present', form_value)
            content = await form_value.read()
            fname = form_value.filename.lower()
            print('content: ', fname)
            try:
                text_content = content.decode("utf-8")
                if form_value.filename.lower().endswith(".txt"):
                    question_text = text_content
                else:
                    text_files[form_value.filename] = text_content
            except UnicodeDecodeError:
                attachments[form_value.filename] = content

                if fname.endswith((".png", ".jpg", ".jpeg")):
                    description = await get_image_description(content)
                    image_descriptions[form_value.filename] = description

    print("question_text:", question_text)
    print("text_files:", list(text_files.keys()))
    print("attachments:", list(attachments.keys()))
    print(image_descriptions)
    # for f in files:
    #     print(f.filename)
    #     content = await f.read()
    #     # Detect text vs binary
    #     try:
    #         text_content = content.decode("utf-8")
    #         if f.filename.lower().endswith(".txt"):
    #             question_text = text_content
    #         else:
    #             text_files[f.filename] = text_content
    #     except UnicodeDecodeError:
    #         # Store binary files as bytes
    #         attachments[f.filename] = content 

    # if file is None and not text:
    #     raise HTTPException(status_code=400, detail="Provide either a file or text form field")

    # if file is not None:
    #     content = await file.read()
    #     question_text = content.decode('utf-8', errors='ignore')
    # else:
    #     question_text = text

    
    

    # 1) Ask LLM for data-fetch code
    fetch_prompt = data_fetch_prompt_template.format(question_text=question_text)
    fetch_prompt = f"""
        Instructions: {fetch_prompt}

        Additional text file contents:
        { {fname: text_files[fname] for fname in text_files} }


        Image descriptions:
        {image_descriptions}

    """
    print('fetching: ' , fetch_prompt)
    print('question:',question_text)
    try:
        fetch_code = call_llm(fetch_prompt, question_text)
    except Exception as e:
        return AnalyzeResponse(success=False, answers=None, error=f"LLM data-fetch failed: {e}")

    # Make sure the model returned code only; if it wrapped in ``` remove fences
    fetch_code = strip_code_fences(fetch_code)

    print('fetch:', fetch_code)
    # 2) Execute fetch code
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    process = multiprocessing.Process(target=_exec_retry_worker, args=(fetch_code, queue))
    process.start()
    process.join(150)  # 150-second timeout

    if process.is_alive():
        process.terminate()
        process.join(2)  # give it 2 seconds to clean up
        if process.is_alive():
            process.kill()  # force kill
        print("took too long")

    if not queue.empty():
        result = queue.get()
        if result["success"]:
            ns = result["result"]
            print('ns:', ns)
        else:
            print(f"Generated code execution failed:\n{result['error']}")
    else:
        try:
            retrydf_prompt2 = f"""
                Instructions: {retrydf_prompt}

                Additional text file contents:
                { {fname: text_files[fname] for fname in text_files} }

            """
            retry_code = call_llm(retrydf_prompt2, question_text)
            print('retrying')
            ns = execute_user_code(retry_code)
            print('ns:', ns)
        except Exception as e:
            return AnalyzeResponse(success=False, answers=None, error=f"Execution of fetch code failed: {e}")
    # print('ns executed', ns)

    # Expect df variable
    if 'df' not in ns:
        return AnalyzeResponse(success=False, answers=None, error="Generated code did not produce a variable named 'df'.")

    df = ns['df']
    if not isinstance(df, pd.DataFrame):
        return AnalyzeResponse(success=False, answers=None, error="'df' is not a pandas DataFrame.")

    # 3) Create a sample representation of the dataframe to pass to LLM
    SAMPLE_ROWS = 100
    sample_df = df.head(SAMPLE_ROWS).copy()
    # provide column types and a small CSV sample
    sample_info = {
        'n_rows': int(len(df)),
        'n_cols': int(df.shape[1]),
        'columns': [{
            'name': str(col),
            'dtype': str(df[col].dtype)
        } for col in df.columns],
        'sample_csv': sample_df.to_csv(index=False),
    }

    sample_info_text = json.dumps(sample_info)
    print('callimg llm for data reading')
    # 4) Ask LLM to generate answer code
    answer_prompt = answer_generation_prompt_template.format(
        question_text=question_text,
        sample_info=sample_info_text
    )
    try:
        answer_code = call_llm(answer_prompt, question_text)
    except Exception as e:
        return AnalyzeResponse(success=False, answers=None, error=f"LLM answer-code generation failed: {e}")

    

    answer_code = strip_code_fences(answer_code)
    print('executing the second code provied', answer_code)
    # 5) Execute answer code in environment where df is present
    try:
        ns2 = execute_user_code(answer_code, provided_env={'df': df})
    except Exception as e:
        return AnalyzeResponse(success=False, answers=None, error=f"Execution of answer code failed: {e}")

    if 'answers' not in ns2:
        return AnalyzeResponse(success=False, answers=None, error="Answer code did not set a variable named 'answers'.")

    answers = ns2['answers']

    # Ensure answers is JSON-serializable; if not, attempt to coerce
    try:
        json_compatible = json.loads(json.dumps(answers, default=_json_serializer))
    except Exception as e:
        return AnalyzeResponse(success=False, answers=None, error=f"Answers not JSON-serializable: {e}")

    return AnalyzeResponse(success=True, answers=json_compatible)

    


# ---------- Utility helpers ----------

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith('```') and s.endswith('```'):
        # remove common fences
        # support ```py or ```python
        first_line_end = s.find('\n')
        if first_line_end != -1:
            # drop first fence line and last fence
            return '\n'.join(s.split('\n')[1:-1]).strip()
        else:
            return s.strip('`')
    # remove single backticks around a short inline code
    if s.startswith('`') and s.endswith('`'):
        return s.strip('`')
    return s


def _json_serializer(obj):
    # fallback serializer for numpy and pandas types and others
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
    except Exception:
        pass
    return str(obj)


# ---------- Entrypoint for local testing ----------
if __name__ == '__main__':
    import uvicorn
    print("Starting Virtual Data Analyst API on http://0.0.0.0:8001")
    uvicorn.run('virtual_data_analyst_api:app', host='0.0.0.0', port=8001, reload=False)
