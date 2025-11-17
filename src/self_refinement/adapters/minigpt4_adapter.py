from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..self_refine_vlm import BaseVLMAdapter


def _has_minigpt4() -> bool:
    return importlib.util.find_spec("minigpt4") is not None


@dataclass
class MiniGPT4Init:
    repo_path: Optional[str] = None
    config_yaml: Optional[str] = None
    ckpt_path: Optional[str] = None
    device: Optional[str] = None
    fallback_on_missing: bool = True


class MiniGPT4Adapter(BaseVLMAdapter):
    """Adapter for MiniGPT-4 model.

    Loads and manages MiniGPT-4 for VLM tasks including answer generation
    and rationale generation.
    """

    def __init__(self, init: MiniGPT4Init):
        self.init = init
        self.ready = False
        self.info = ""
        self.backend = None
        self.vis_processor = None
        self.chat = None
        self.device_id = 0

        if init.repo_path:
            repo = Path(init.repo_path)
            if repo.exists():
                sys.path.insert(0, str(repo))

        if _has_minigpt4():
            try:
                import torch
                from transformers import StoppingCriteriaList
                from minigpt4.common.config import Config
                from minigpt4.common.registry import registry
                from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, StoppingCriteriaSub
                
                # Determine device
                if init.device:
                    self.device_id = int(init.device.split(':')[-1]) if ':' in init.device else 0
                
                # Load configuration
                config_path = init.config_yaml or "eval_configs/minigptv2_eval.yaml"
                
                # Create args object with proper attributes
                class Args:
                    def __init__(self, cfg_path, gpu_id):
                        self.cfg_path = cfg_path
                        self.options = []
                        self.gpu_id = gpu_id
                
                args = Args(cfg_path=config_path, gpu_id=self.device_id)
                cfg = Config(args)
                
                # Load model
                model_config = cfg.model_cfg
                model_config.device_8bit = self.device_id
                model_cls = registry.get_model_class(model_config.arch)
                model = model_cls.from_config(model_config).to(f'cuda:{self.device_id}')
                
                # Load visual processor
                vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
                vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
                
                # Setup stopping criteria
                stop_words_ids = [[835], [2277, 29937]]
                stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{self.device_id}') for ids in stop_words_ids]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
                
                # Create chat interface
                chat = Chat(model, vis_processor, device=f'cuda:{self.device_id}', stopping_criteria=stopping_criteria)
                
                self.backend = model
                self.vis_processor = vis_processor
                self.chat = chat
                self.ready = True
                self.info = "MiniGPT-4 model loaded successfully."
                print(f"[MiniGPT4Adapter] {self.info}")
                
            except Exception as e:
                import traceback
                self.info = f"Failed to load MiniGPT-4: {str(e)}"
                print(f"[MiniGPT4Adapter] {self.info}")
                print(traceback.format_exc())
                if not init.fallback_on_missing:
                    raise RuntimeError(self.info)
        else:
            self.info = (
                "MiniGPT-4 not found. Using placeholder outputs (fallback_on_missing)."
            )
            if not init.fallback_on_missing:
                raise RuntimeError(
                    "MiniGPT-4 package not installed. Install your MiniGPT-4 repo (pip install -e .) or enable fallback_on_missing."
                )

    # --- BaseVLMAdapter interface ---
    def generate_answer(self, image: Any, question: str, **gen_kwargs) -> str:
        if self.ready and self.chat:
            try:
                from minigpt4.conversation.conversation import CONV_VISION_LLama2
                
                # Create a conversation state
                chat_state = CONV_VISION_LLama2.copy()
                img_list = []
                
                # Upload and encode image
                self.chat.upload_img(image, chat_state, img_list)
                self.chat.encode_img(img_list)
                
                # Ask question and get answer
                self.chat.ask(question, chat_state)
                
                # Get response
                num_beams = gen_kwargs.get('num_beams', 1)
                temperature = gen_kwargs.get('temperature', 1.0)
                max_new_tokens = gen_kwargs.get('max_new_tokens', 300)
                
                answer = self.chat.answer(conv=chat_state,
                                         img_list=img_list,
                                         num_beams=num_beams,
                                         temperature=temperature,
                                         max_new_tokens=max_new_tokens,
                                         max_length=2000)[0]
                return answer.strip()
            except Exception as e:
                print(f"Error generating answer: {e}")
                # Fallback to stub
                q = (question or "").strip().rstrip("?")
                return f"[MiniGPT-4] Answer to: {q}"
        
        # Fallback deterministic output
        q = (question or "").strip().rstrip("?")
        return f"[MiniGPT-4 stub] Answer to: {q}"

    def generate_rationale(self, image: Any, question: str, answer: str, **gen_kwargs) -> str:
        if self.ready and self.chat:
            try:
                from minigpt4.conversation.conversation import CONV_VISION_LLama2
                
                # Create a conversation state
                chat_state = CONV_VISION_LLama2.copy()
                img_list = []
                
                # Upload and encode image
                self.chat.upload_img(image, chat_state, img_list)
                self.chat.encode_img(img_list)
                
                # Ask for rationale
                rationale_prompt = f"Question: {question}\nAnswer: {answer}\nProvide a brief rationale for this answer:"
                self.chat.ask(rationale_prompt, chat_state)
                
                # Get response
                num_beams = gen_kwargs.get('num_beams', 1)
                temperature = gen_kwargs.get('temperature', 1.0)
                max_new_tokens = gen_kwargs.get('max_new_tokens', 200)
                
                rationale = self.chat.answer(conv=chat_state,
                                            img_list=img_list,
                                            num_beams=num_beams,
                                            temperature=temperature,
                                            max_new_tokens=max_new_tokens,
                                            max_length=2000)[0]
                return rationale.strip()
            except Exception as e:
                print(f"Error generating rationale: {e}")
                # Fallback to stub
                return f"[MiniGPT-4] Rationale: The image cues support the answer '{answer}'."
        
        return f"[MiniGPT-4 stub] Rationale: The image cues support the answer '{answer}'."

