"""
Market Anomaly Detector - Configuration
========================================
API keys and client management
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class APIConfig:
    """API 키 및 클라이언트 관리"""
    
    _clients = {}
    
    # API 키 환경변수
    KEYS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'perplexity': 'PERPLEXITY_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'fred': 'FRED_API_KEY'
    }
    
    @classmethod
    def get_key(cls, api: str) -> Optional[str]:
        """API 키 반환"""
        env_var = cls.KEYS.get(api.lower())
        if env_var:
            return os.getenv(env_var)
        return None
    
    @classmethod
    def validate(cls) -> Dict[str, bool]:
        """모든 API 키 상태 확인"""
        return {api: bool(cls.get_key(api)) for api in cls.KEYS}
    
    @classmethod
    def get_client(cls, api: str):
        """API 클라이언트 반환 (싱글톤)"""
        api = api.lower()
        
        if api in cls._clients:
            return cls._clients[api]
        
        key = cls.get_key(api)
        if not key:
            raise ValueError(f"{api.upper()} API key not found")
        
        if api == 'openai':
            from openai import OpenAI
            cls._clients[api] = OpenAI(api_key=key)
        
        elif api == 'anthropic':
            import anthropic
            cls._clients[api] = anthropic.Anthropic(api_key=key)
        
        elif api == 'perplexity':
            from openai import OpenAI
            cls._clients[api] = OpenAI(
                api_key=key,
                base_url="https://api.perplexity.ai"
            )
        
        elif api == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=key)
            cls._clients[api] = genai.GenerativeModel('gemini-pro')
        
        else:
            raise ValueError(f"Unknown API: {api}")
        
        return cls._clients[api]


# 모델 설정
MODELS = {
    'orchestrator': 'gpt-4o',
    'code_gen': 'claude-sonnet-4-20250514',
    'analysis': 'claude-sonnet-4-20250514',
    'search': 'sonar-pro',
    'summary': 'claude-sonnet-4-20250514'
}

# 에이전트 설정
AGENT_CONFIG = {
    'orchestrator': {
        'model': MODELS['orchestrator'],
        'max_tokens': 4000,
        'temperature': 0.3
    },
    'claude': {
        'model': MODELS['code_gen'],
        'max_tokens': 8000,
        'temperature': 0.2
    },
    'perplexity': {
        'model': MODELS['search'],
        'max_tokens': 4000,
        'temperature': 0.1
    }
}

# 프로젝트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
