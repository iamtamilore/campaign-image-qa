import os, sys, time, json, requests
from pathlib import Path

# The call log is research evidence, so it is vendored into the repo rather than
# expected to live on one machine. If it cannot be imported the pipeline still
# runs; only the log line is lost.
try:
    from prompt_logger import log_request, log_result
except ImportError:  # pragma: no cover
    def log_request(*a, **k):
        pass

    def log_result(*a, **k):
        pass

#the main address where kie.ai lives
BASE_URL = "https://api.kie.ai"


class HeroGenerator:
    def __init__(self, api_key: str | None = None):
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

    #UPLOAD
    #take file path(str), return a url (str)
    def upload(self, local_path: str) -> str:
        "upload local file to catbox.moe (3 tries), litterbox as backup, return public URL"
        print(f"[upload] {local_path}")

        #try catbox first, 3 attempts, wait max 60 seconds each
        for attempt in range(3):
            try:
                #open file from path, read as binary(not text), call it f while we use it
                with open(local_path, "rb") as f:
                    #send request to catbox website, tell catbox "upload mode"
                    r = requests.post("https://catbox.moe/user/api.php",
                                      data={"reqtype": "fileupload"},
                                      files={"fileToUpload": f}, timeout=60)
                #did website say 'yes', and does the response start with http (a real url)
                if r.ok and r.text.strip().startswith("http"):
                    #save the url catbox gave us
                    url = r.text.strip()
                    print(f"[upload] catbox → {url}")
                    return url
                print(f"[upload] catbox attempt {attempt+1} bad response: {r.text[:80]}")
            except Exception as e:
                print(f"[upload] catbox attempt {attempt+1} failed: {e}")
            #wait 3 seconds before trying again (not on the last attempt)
            if attempt < 2:
                time.sleep(3)

        #catbox failed all 3 tries, fall back to litterbox (same site, files expire in 72h)
        print("[upload] catbox failed, trying litterbox (72h expiry)")
        try:
            with open(local_path, "rb") as f:
                r = requests.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "72h"},
                    files={"fileToUpload": f}, timeout=60)
            if r.ok and r.text.strip().startswith("http"):
                url = r.text.strip()
                print(f"[upload] litterbox → {url}")
                return url
        except Exception as e:
            print(f"[upload] litterbox failed: {e}")

        #both hosts said no, show something broke
        raise RuntimeError(f"All upload hosts failed for {local_path}")

    #CREATE TASK
    #tell kie.ai to start a task, take instructions as dict, return a task id (str)
    def _post(self, payload: dict) -> str:
        #call kie.ai's server, show auth key/headers, change instructions to json, wait max 30 seconds
        r = requests.post(f"{BASE_URL}/api/v1/jobs/createTask",
                          headers=self.headers, data=json.dumps(payload), timeout=30)
        #if kie.ai says no, show the error code and first 400 chars of why
        if not r.ok:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:400]}")
        #change response to a python dictionary
        resp = r.json()
        data = resp.get("data")
        #sometimes kie.ai answers ok but sends empty data, catch that
        if not data:
            raise RuntimeError(f"API null data, full response: {resp}")
        return data["taskId"]

    #order an image from gpt-image-2, keeping the face from my reference photos
    def create_gpt_i2i_task(self, prompt: str, ref_urls: list[str]) -> str:
        "gpt-image-2-image-to-image, face-lock + dressing, costs ~6 credits. keep prompts MINIMAL, aggressive negatives cause drift"
        payload = {
            "model": "gpt-image-2-image-to-image",
            "input": {
                #what i want in the image
                "prompt":       prompt,
                #never low or medium, low = plastic skin
                "quality":      "high",
                #portrait, tall shape
                "aspect_ratio": "3:4",
                #my reference pictures
                "input_urls":   ref_urls,
            }
        }
        #send the task to kie.ai, get back a task ID
        task_id = self._post(payload)
        print(f"[gpt-image-2-i2i] task → {task_id}  (~6cr)")
        log_request(task_id, payload["model"], "gpt_i2i", prompt,
                    input_urls=ref_urls, input_params=payload["input"])
        return task_id

    #order an image from flux-2, my main model for face lock + angle variants
    def create_flux_angle_task(self, prompt: str, ref_url: str,
                                negative: str = "") -> str:
        "flux-2/pro-image-to-image, hero lock / angle variants, costs ~5 credits"
        payload = {
            "model": "flux-2/pro-image-to-image",
            "input": {
                #what i want in the image
                "prompt":          prompt,
                #my reference picture
                "input_urls":      [ref_url],
                #portrait, tall shape
                "aspect_ratio":    "3:4",
                #resolution, 1000 pixels
                "resolution":      "1K",
                #turn off adult-content filter
                "nsfw_checker":    False,
                #these are exactly what i dont want in the image
                "negative_prompt": negative,
            }
        }
        #send the task to kie.ai, get back a task ID
        task_id = self._post(payload)
        print(f"[flux-2] task → {task_id}  (~5cr)")
        log_request(task_id, payload["model"], "flux_angle", prompt,
                    input_urls=[ref_url], input_params=payload["input"])
        return task_id

    #DRESS SUBJECT IN A REAL GARMENT, Condition B's validated recipe (Rule 9)
    #subject wearing an exact real garment (with legible branding/logo), locked
    #to the exact reference-image setup that measured best: subject ref + garment
    #ref, plain prompt, wide framing so the garment isn't cropped. Nothing more.
    def create_garment_dress_task(self, identity_core: str, subject_ref_url: str,
                                   garment_ref_url: str) -> str:
        "flux-2/pro-image-to-image, subject wearing a real referenced garment, ~5cr. DO NOT add extra crops/swatches/color language to this prompt -- measured worse every time (M16.13)."
        prompt = (
            f"{identity_core}\n"
            "Wearing the exact garment shown in the second reference image, "
            "including any text, wordmark and trim, unchanged. "
            "Half-body fashion shot, camera pulled back, framed from mid-thigh "
            "to just above the head so the ENTIRE garment is visible including "
            "any branding and the hem -- do NOT crop the garment, do NOT do a "
            "tight head-and-shoulders portrait crop.\n"
            "Gray background, one image only."
        )
        negative = ("tight headshot crop, close-up portrait, cropped torso, "
                    "cropped logo text, cropped garment hem")
        payload = {
            "model": "flux-2/pro-image-to-image",
            "input": {
                "prompt":          prompt,
                #subject first, garment second -- matches the tested/measured order
                "input_urls":      [subject_ref_url, garment_ref_url],
                "aspect_ratio":    "3:4",
                "resolution":      "1K",
                "nsfw_checker":    False,
                "negative_prompt": negative,
            }
        }
        task_id = self._post(payload)
        print(f"[flux-2 dress] task → {task_id}  (~5cr)")
        log_request(task_id, payload["model"], "garment_dress", prompt,
                    input_urls=[subject_ref_url, garment_ref_url],
                    input_params=payload["input"])
        return task_id

    #POLL + DOWNLOAD
    #keep asking kie.ai "is it done yet?" every 5 seconds, give up after 180 seconds
    def poll(self, task_id: str, poll_every: int = 5, timeout: int = 180) -> dict:
        #calculate the time we will give up: now + 180 seconds
        deadline = time.time() + timeout
        #keep looping as long as we havent hit the deadline
        while time.time() < deadline:
            #ask kie.ai for the status of my task
            r = requests.get(f"{BASE_URL}/api/v1/jobs/recordInfo",
                             headers=self.headers,
                             params={"taskId": task_id}, timeout=30)
            #crash if the server answers with an error
            r.raise_for_status()
            data  = r.json()["data"]
            #pull out the state (waiting/success/fail), lowercase it
            state = data.get("state", "unknown").lower()
            print(f"[kie] {task_id[:18]}...  {state}  ({data.get('creditsConsumed','?')} cr)")
            #done? give back the full result
            if state == "success": return data
            #failed? show why and stop
            if state in ("fail","failed"):
                raise RuntimeError(f"Task failed: {data.get('failMsg')} / {data.get('failCode')}")
            #not done yet, wait 5 seconds then ask again
            time.sleep(poll_every)
        raise TimeoutError(f"Timed out after {timeout}s")

    #pull the image url out of the finished result and save it to disk
    def download(self, result: dict, out_path: Path) -> Path:
        #the result comes as json-inside-json, unwrap it (or empty dict if missing)
        result_json = json.loads(result.get("resultJson") or "{}")
        #the url can live under two different names depending on the model
        urls = result_json.get("resultUrls") or result_json.get("images") or []
        if not urls:
            raise ValueError(f"No image URL in result:\n{result}")
        #download the first image, wait max 60 seconds
        r = requests.get(urls[0], timeout=60)
        r.raise_for_status()
        #create the output folder if it dont exist yet
        out_path.parent.mkdir(parents=True, exist_ok=True)
        #write the raw image bytes to disk
        out_path.write_bytes(r.content)
        print(f"[kie] saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
        return out_path

    #shortcut: wait for the task then download it, one call
    def run(self, task_id: str, out_path: Path) -> Path:
        t0 = time.time()
        try:
            result = self.poll(task_id)
        except Exception as e:
            log_result(task_id, "fail", fail_reason=str(e),
                       duration_seconds=time.time() - t0)
            raise
        saved = self.download(result, out_path)
        result_json = json.loads(result.get("resultJson") or "{}")
        urls = result_json.get("resultUrls") or result_json.get("images") or []
        log_result(task_id, "success",
                   credits_consumed=result.get("creditsConsumed"),
                   output_url=urls[0] if urls else None,
                   local_output_path=str(saved),
                   duration_seconds=time.time() - t0)
        return saved

    #IDENTITY SPEC + ANGLE PROMPTS

    @staticmethod
    def load_identity_spec(subject_data_dir: Path) -> str:
        "read the BASE PROMPT CORE out of IDENTITY_SPEC.txt, or a simple fallback if missing"
        spec_path = subject_data_dir / "IDENTITY_SPEC.txt"
        #no spec file? use the minimal face-lock line
        if not spec_path.exists():
            return (
                "Keep the face from the reference image exactly, "
                "identical face, hair, skin tone and features."
            )
        text = spec_path.read_text()
        marker = "BASE PROMPT CORE"
        #spec file exists but has no core section? same fallback
        if marker not in text:
            return "Keep the face from the reference image exactly."

        #take everything after the marker
        after = text.split(marker, 1)[1]
        #the core sits between the first and last quote marks
        start = after.find('"')
        end   = after.rfind('"')
        if start == -1 or start == end:
            return "Keep the face from the reference image exactly."

        raw = after[start + 1 : end]
        #drop the placeholder line, the angle instruction gets added later
        lines = [l for l in raw.splitlines() if "[ANGLE INSTRUCTION HERE]" not in l]
        return "\n".join(lines).strip()

    #glue identity + angle instruction + anti double-shot warning into one prompt
    def _angle_prompt(self, identity_core: str, angle_instruction: str) -> str:
        "identity FIRST (model weights earlier terms more), then the angle, then the do-NOTs"
        return (
            f"{identity_core}\n"
            f"{angle_instruction}\n"
            "Do NOT generate a side-by-side comparison, "
            "do NOT generate two views in one frame, do NOT generate a turnaround."
        )

    #the 5 angle shots, in the order that keeps the face consistent
    def get_angle_jobs(self, identity_core: str) -> list[tuple[str, str]]:
        "return (filename, prompt) for all 5 angles. order: 3/4 left → 3/4 right → profile left → profile right → back. wider shot = less face pressure = less drift"
        return [
            #3/4 angles FIRST, least face pressure, establish identity before demanding profiles
            ("hero_3qleft.png",
             self._angle_prompt(identity_core,
                "Three-quarter view. Head turned approximately 45 degrees to the LEFT. "
                "Both eyes visible. Left side of face dominant in frame. Jaw relaxed, unbothered.")),

            ("hero_3qright.png",
             self._angle_prompt(identity_core,
                "Three-quarter view. Head turned approximately 45 degrees to the RIGHT. "
                "Both eyes visible. Right side of face dominant in frame. Jaw relaxed, unbothered.")),

            #profile shots, camera-POSITION language, head-direction language causes double-shot fails
            ("hero_profileleft.png",
             self._angle_prompt(identity_core,
                "The camera is positioned directly to the LEFT of the subject, "
                "aimed at the LEFT ear and LEFT cheek. "
                "Subject is looking straight ahead, not at the camera. "
                "Left side of the face fully visible, right side completely hidden behind the head. "
                "Pure left profile. ONE person, ONE angle.")),

            ("hero_profileright.png",
             self._angle_prompt(identity_core,
                "The camera is positioned directly to the RIGHT of the subject, "
                "aimed at the RIGHT ear and RIGHT cheek. "
                "Subject is looking straight ahead, not at the camera. "
                "Right side of the face fully visible, left side completely hidden behind the head. "
                "Pure right profile. ONE person, ONE angle.")),

            #back, zero face visible, no consistency pressure at all
            ("hero_back.png",
             self._angle_prompt(identity_core,
                "Back view. Camera is directly behind the subject. "
                "Subject is facing completely away from the camera. "
                "Back of neck, hair, and upper shoulders in frame. Zero face visible.")),
        ]

    #FULL CHARACTER SHEET FROM SEED, the method the project actually ran

    def generate_character_sheet_from_seed(self, subject_id: str,
                                            seed_path: str,
                                            out_dir: Path) -> dict:
        "pinterest seed → hero_front via BOTH models (flux-2 + gpt-2) so i can compare them. angles are done separately by generate_angle_comparison.py"
        #make the output folders, one per model so results dont mix
        out_dir.mkdir(parents=True, exist_ok=True)
        flux_dir  = out_dir / "flux-2-test"
        gpt2_dir  = out_dir / "gpt-2-test"
        flux_dir.mkdir(exist_ok=True)
        gpt2_dir.mkdir(exist_ok=True)

        #hero_front goes inside its model folder, not in the root
        front_path      = flux_dir / "hero_front.png"
        gpt2_front_path = gpt2_dir / "hero_front.png"

        print(f"\n=== Generating character sheet for {subject_id} (seeded from {seed_path}) ===")

        #the face-lock prompt: keep the reference face exactly, clean grey studio
        front_prompt = (
            "Keep the face from the reference image exactly, identical face, "
            "skin tone, bone structure and features. "
            "Front-facing portrait, head straight to camera, neutral expression. "
            "Neutral grey studio background, even soft diffused lighting. "
            "Natural skin texture, visible pores, no retouching, no plastic skin. "
            "Portrait orientation. ONE image only."
        )

        #upload seed once, both models use the same url
        seed_url = self.upload(seed_path)

        #step 1a: flux-2 front → flux-2-test/hero_front.png
        if front_path.exists():
            #re-run safe: dont regenerate (and dont pay) for what already exists
            print(f"[skip] flux-2-test/hero_front.png exists")
        else:
            print(f"\n[hero] flux-2 front from seed...")
            try:
                t = self.create_flux_angle_task(front_prompt, seed_url)
                self.run(t, front_path)
            except Exception as e:
                print(f"[hero] flux-2 front failed: {e}")

        #step 1b: gpt-2 front → gpt-2-test/hero_front.png (the comparison shot)
        if gpt2_front_path.exists():
            print(f"[skip] gpt-2-test/hero_front.png exists")
        else:
            print(f"\n[hero] gpt-2 front from seed (comparison)...")
            try:
                t = self.create_gpt_i2i_task(front_prompt, [seed_url])
                self.run(t, gpt2_front_path)
            except Exception as e:
                #non-fatal: the flux front is the main one, gpt-2 is just for comparing
                print(f"[hero] gpt-2 front failed (non-fatal): {e}")

        #summary, just the two fronts (angles done separately via generate_angle_comparison.py)
        flux_ok = front_path.exists()
        gpt2_ok = gpt2_front_path.exists()
        print(f"\n=== {subject_id} fronts ===")
        print(f"  flux-2-test/hero_front.png  {'✓' if flux_ok else 'MISSING'}")
        print(f"  gpt-2-test/hero_front.png   {'✓' if gpt2_ok else 'MISSING'}")
        print(f"\nNext: python generate_angle_comparison.py --subject {subject_id[-1]} --n 5")

        saved = int(flux_ok) + int(gpt2_ok)
        return {"subject": subject_id, "saved": saved, "missing": [] if saved == 2 else ["hero_front"]}
