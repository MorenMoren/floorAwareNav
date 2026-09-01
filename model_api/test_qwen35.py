import numpy as np
import random
from openai import OpenAI
import os
import json
import re
import base64
import cv2
# os.environ["OUT_HOST"] = 

class QWen3_5:
    """QWen3_5 Image-Text Matching model using transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3.6-35B-A3B") -> None:
        """
        初始化QWen3.5模型。

        :param model_name: 模型名称或路径
        """
        self.model_name = os.environ.get("ASCENT_QWEN_MODEL", model_name)
        self.base_url = os.environ.get(
            "ASCENT_QWEN_BASE_URL", "http://172.16.33.7:11435/v1"
        )
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

    @staticmethod
    def _image_data_url(image: np.ndarray) -> str:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        if not ok:
            raise ValueError("failed to encode VLM image")
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

    def chat(self, txt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an AI assistant with advanced spatial reasoning capabilities. Your task is to choose the optimal option to find the target object.you just need to give a brief reason and only one valid json answer"},
            {"role": "user", "content": txt}
        ]
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=10240,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )

        return chat_response.choices[0].message.content

    def chat_with_image(self, txt: str, image: np.ndarray) -> str:
        return self.chat_with_images(txt, [image])

    def chat_with_images(self, txt: str, images: list[np.ndarray]) -> str:
        content = [{"type": "text", "text": txt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_url(image)},
            }
            for image in images
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return only the requested JSON."},
                {"role": "user", "content": content},
            ],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content or ""


def extract_json_from_response(clean_text: str) -> dict:
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass # 清理 Markdown 后依然失败，进入终极策略

    # 策略 3: 终极正则提取 (寻找最外层的大括号 {} )
    # 这个正则会匹配第一个 { 和最后一个 } 之间的所有内容，忽略前后的废话
    # DOTALL 确保可以匹配换行符
    match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
    
    if match:
        extracted_str = match.group(1)
        try:
            return json.loads(extracted_str)
        except json.JSONDecodeError as e:
            # 找到了大括号，但是里面的内容不是合法的 JSON (比如逗号遗漏、转义错误等)
            raise ValueError(f"提取到了类似 JSON 的结构，但语法有误: {e}\n提取出的字符串:\n{extracted_str}")
            
    # 如果以上所有策略都失败了
    raise ValueError("无法从大模型的回复中找到任何有效的 JSON 结构。\n原始回复:\n" + clean_text)



# if __name__ == "__main__":
#     #model = QWen3_5()
#     #response = model.chat("Give me a short introduction to large language models.")
#     #print("Response:", response)
    
#     clean_text = '''{
#     "Index": "2",
#     "Reason": "While the prior probability for a toilet in a Bathroom is highest (97.0%), no bathroom is listed in the area descriptions. Among the available options, Area 2 (Hall) has the highest prior probability (0.2%) compared to Area 1 (Kitchen, 0.5% - wait, Kitchen is 0.5% which is higher than Hall's 0.2%). Let me re-evaluate based on the provided probabilities: Kitchen (0.5%) > Hall (0.2%) > Living Room (0.1%). However, looking at the object lists, Area 1 (Kitchen) contains 'kitchen', 'kitchen island', 'oven', etc., which are very specific to a kitchen. Area 2 (Hall) contains 'hallway', 'doorway', 'lead to', which are transitional spaces. Area 3 (Living room) contains 'couch', 'slide', 'stair'. \n\nActually, let's look at the probabilities again. \nKitchen: 0.5%\nHall: 0.2%\nLiving_room: 0.1%\n\nBased strictly on the provided prior probabilities, the Kitchen (Area 1) has the highest probability (0.5%) among the available areas. Although 0.5% is very low, it is the highest relative to the other options provided (0.2% and 0.1%). Therefore, Area 1 is the optimal choice based on the data."
# }

# Wait, I need to be careful. The prompt asks for the *optimal* area based on *prior probabilistic data*.
# The probabilities are:
# Kitchen: 0.5%
# Hall: 0.2%
# Living_room: 0.1%

# Area 1 is a Kitchen.
# Area 2 is a Hall.
# Area 3 is a Living Room.

# Comparing the probabilities:
# Area 1 (Kitchen) = 0.5%
# Area 2 (Hall) = 0.2%
# Area 3 (Living Room) = 0.1%

# The highest probability among the available areas is Area 1.
# '''
#     clean_text = clean_text.replace("\n", "").replace("\r", "")
#     extracted_json = extract_json_from_response(clean_text)
#     print("Extracted JSON:", extracted_json)
#     llm = QWen3_5()
#     response = llm.chat(extracted_json["Reason"])
#     print("Response:", response)
