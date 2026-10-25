import os, sys, time, json, requests
from pathlib import Path

# Vendored call log. If it cannot be imported the client still works; only the
# log line is lost.
try:
    from prompt_logger import log_request
except ImportError:  # pragma: no cover
    def log_request(*a, **k):
        pass

#The main address where Kie.ai lives
BASE_URL = "https://api.kie.ai"
#catbox holds files temporarily for free
CATBOX   = "https://catbox.moe/user/api.php"


class KieClient:
    def __init__(self, api_key: str | None = None, out_dir: str = "outputs/generated"):
        #check sys env for api key
        self.api_key = api_key or os.getenv("KIE_API_KEY", "")
        #error fallback
        if not self.api_key:
            raise EnvironmentError(
                "KIE_API_KEY not set.\n")
        #every call carries; api-key, as well as data/application in json format
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        #turn out_dir to a path object for easy integration across mac, windows
        self.out_dir = Path(out_dir)
        #if it dont exist, create the folder, create parent folders, if it exists already, if folder exists already OK
        self.out_dir.mkdir(parents=True, exist_ok=True)

    #UPLOAD
    #take file path(str), return a url (str)
    def upload(self, local_path: str) -> str:
        "upload local file to catbox.moe and return public URL"
        print(f"[kie] uploading {local_path}")
        #open files from path, read as binary(not text), call it f while we use it
        with open(local_path, "rb") as f:
            #send requests to catbox website, tell catbox "upload mode"
            r = requests.post(CATBOX, data={"reqtype": "fileupload"},
                              #file to upload is object f, wait max 30 seconds
                              files={"fileToUpload": f}, timeout=30)
        #check if upload succeded, did website say 'yes',ok?, remove blank spaces from the response, and check if it sarts with http 
        if r.ok and r.text.strip().startswith("http"):
            #save the url the catbox gave us 
            url = r.text.strip()

            print(f"[kie] uploaded, check url {url}")
            return url
        #show something broke, print out what catbox said
        raise RuntimeError(f"Upload failed: {r.text}")
    #tell kieai, to start a task, take instrcutions as dict, return a task id as (str)
    def _create_task(self, payload: dict) -> str:
        #call kie.ai's server/base_url, take full address of the task, show auth key/headers, change instructions to json, wait max 30 seconds
        r = requests.post(f"{BASE_URL}/api/v1/jobs/createTask",
                          headers=self.headers, data=json.dumps(payload), timeout=30)
        # if kie.ai crashes, show why
        r.raise_for_status()
        #change response to a python dictionary, find task id, inside "data" column
        task_id = r.json()["data"]["taskId"]
        return task_id
#order a hero image from kie.ai
    def create_hero_task(self, prompt: str, input_urls: list[str],
                         negative_prompt: str = "") -> str:
        "flux-2/pro-image-to-image, i.e converting our pinterest reference photo to extract subject id, and lock as hero image cost; 5 credits. Face + skin lock. Max 8"
        payload = {
            "model": "flux-2/pro-image-to-image",
            "input": {
                #take what i want you to do
                "prompt": prompt,
                #reference pictures
                "input_urls": input_urls,
                #portrait, tall shape
                "aspect_ratio": "3:4",
                #resolution shape, 1000 pixels
                "resolution": "1K",
                #turn off adult-content filter
                "nsfw_checker": False,
                #this are exactly what i dont want in the image
                "negative_prompt": negative_prompt or
                    "stud earring, cartilage piercing, plastic skin, airbrushed, blurry",
            }
        }
        #send the task to kia.ai, get back a task ID
        task_id = self._create_task(payload)
        print(f"[kie] flux-2 hero task, {task_id}, costs around (~5 credits)")
        log_request(task_id, payload["model"], "hero_task", prompt,
                    input_urls=input_urls, input_params=payload["input"])
        return task_id