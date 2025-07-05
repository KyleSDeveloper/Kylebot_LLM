import json
import datetime

# Simple task storage
tasks = {"tasks": []}

def save_task(task, due_date):
    tasks["tasks"].append({"task": task, "due": due_date})
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)
    return f"Added task: {task} by {due_date}"

def list_tasks():
    with open("tasks.json", "r") as f:
        tasks = json.load(f)
    return "\n".join([f"- {t['task']} (Due: {t['due']})" for t in tasks["tasks"]])

# Basic chatbot loop
while True:
    user_input = input("Yo Kyle, what's up? ")
    if "add task" in user_input.lower():
        task = user_input.replace("add task", "").strip()
        due = str(datetime.date.today())  # Placeholder
        print(save_task(task, due))
    elif "list tasks" in user_input.lower():
        print(list_tasks())
    elif "quit" in user_input.lower():
        print("Catch ya later!")
        break
    else:
        print("Huh? Try 'add task', 'list tasks', or 'quit'.")