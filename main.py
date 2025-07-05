import json
import datetime
import requests
import os
from dotenv import load_dotenv


# Load or initialize tasks
try:
    with open("tasks.json", "r") as f:
        tasks = json.load(f)
except FileNotFoundError:
    tasks = {"tasks": []}

# Load API key from .env file
load_dotenv()
api_key = os.getenv('OPENWEATHERMAP_API_KEY')

def save_task(task, due_date):
    tasks["tasks"].append({"task": task, "due": due_date})
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)
    return f"Locked in, Kyle: {task} by {due_date}"

def list_tasks():
    if not tasks["tasks"]:
        return "You’re chillin’, no tasks!"
    return "\n".join([f"- {t['task']} (Due: {t['due']})" for t in tasks["tasks"]])

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return f"Oof, couldn’t find {city}. Try again?"
        desc = response["weather"][0]["description"]
        temp = response["main"]["temp"]
        return f"Weather in {city}: {desc}, {temp}°C"
    except Exception as e:
        return f"Yo, weather’s acting up: {str(e)}"

# Kylebot loop
print("Kylebot v0.1.0 - Your Personal Wingman")
while True:
    user_input = input("Yo, what’s the vibe? ")
    if "add task" in user_input.lower():
        task = user_input.replace("add task", "").strip()
        due = str(datetime.date.today())
        print(save_task(task, due))
    elif "list tasks" in user_input.lower():
        print(list_tasks())
    elif "weather" in user_input.lower():
        city = user_input.replace("weather", "").strip()
        if not city:
            city = "Chicago"  # Default city, change as needed
        print(get_weather(city))
    elif "quit" in user_input.lower():
        print("Later, my dude!")
        break
    else:
        print("Bruh, try 'add task <task>', 'list tasks', 'weather <city>', or 'quit'.")