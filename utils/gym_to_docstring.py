import os
import ast

def gym_docstring_check(env: str, verbose: bool = False):
    """
    Check if each action name (e.g., "CountOccurrences") in self.func_mapping defined in __init__ of the environment file specified by env_path exists and has a docstring.
    Whether the method corresponding to each action name (e.g., "CountOccurrences") exists and has a docstring.

    Assumption: action name == method name (consistent with your example environment).
    """
    try:
        if not os.path.exists(env):
            tree = ast.parse(env)
        else:
            with open(env, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), env)
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to parse file {env}: {e}")
        return False, {}

    # Get the first class (usually the environment class)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if cls is None:
        if verbose:
            print("No class definition found.")
        return False, {}

    # Collect methods -> docstring
    methods = {n.name: ast.get_docstring(n, clean=False) for n in cls.body if isinstance(n, ast.FunctionDef)}

    # Find __init__
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None:
        if verbose:
            print("No __init__ method found.")
        return False, {}

    # Find self.func_mapping = {...} in __init__
    action_labels = []
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr == "func_mapping"
                ):
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                action_labels.append(k.value)

    # Check if the docstring of the method corresponding to each action name exists
    results = {}
    flag = True
    if verbose:
        print("=== Docstring check for env: ===")

    for label in action_labels:
        doc = methods.get(label, None)
        if doc is None:
            if verbose:
                print(f"[ERROR] {label}: method not found.")
            results[label] = "NO_METHOD"
            flag = False
        elif doc.strip():
            if "Args" not in doc or "Returns" not in doc or "Example" not in doc:
                if verbose:
                    print(f"[WARN]  {label}: missing args and returns.")
                results[label] = "NO_ARGS_RETURNS"
                flag = False
            else:
                if verbose:
                    print(f"[OK]    {label}: has docstring.")
                results[label] = doc
        else:
            if verbose:
                print(f"[WARN]  {label}: missing docstring.")
            results[label] = "NO_DOC"
            flag = False

    if flag:
        if verbose:
            print("=== Docstring check passed. ===")
    else:
        if verbose:
            print("=== Docstring check failed. ===")
    return flag, results

def format_docstring(results: dict):
    docstring_prompt = ""
    for label, doc in results.items():
        docstring_prompt += f"Function:\ndef {label}:\n{doc}\n"
    return docstring_prompt