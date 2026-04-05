import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import warnings
import time
import random

# 忽略 SSL 警告
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# [新增] 动态指纹池，用于 _request_with_jitter 轮换
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/119.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
]

class NetworkClient:
    """
    [底层网络层 - 终极增强版 V417]
    1. 升级 User-Agent 为 PC 浏览器，解决移动端限流问题。
    2. [并发修复] 扩容 HTTPAdapter 底层连接池 (pool_maxsize=50)，彻底根除多线程饿死。
    3. 完美保留 SSL 忽略、Keep-Alive 复用与自动 Retry 机制。
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False
        
        # [核心增强] 扩容连接池到 50，满足我们 32 线程的暴力并发需求！
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(
            max_retries=retries, 
            pool_connections=50,  # 允许保持的最大连接池数量
            pool_maxsize=50       # 允许的最大并发连接数
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # 升级为 PC 端 Header (严格保留原版字典定义)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "http://quote.eastmoney.com/",
            "Accept": "*/*",
            "Connection": "keep-alive"
        }

    def _request_with_jitter(self, method, url, max_attempts=3, **kwargs):
        """[新增核心防拉黑机制] 带指数退避与随机抖动的安全请求"""
        for attempt in range(max_attempts):
            try:
                # [修复 4] 增加 or self.headers 兜底，防止外部传入 headers=None 导致 AttributeError
                req_headers = (kwargs.get('headers') or self.headers).copy()
                req_headers['User-Agent'] = random.choice(USER_AGENTS)
                kwargs['headers'] = req_headers
                
                # 强制加入 verify=False 以匹配你原版的所有请求形态
                kwargs['verify'] = False
                
                resp = self.session.request(method, url, **kwargs)
                
                # [修复 1] 放宽成功状态码判定，囊括 201-299，防止成功请求被静默丢弃
                if 200 <= resp.status_code < 300:
                    return resp
                
                # 若触发业务层面的拦截限频，执行外层的随机抖动与退避
                elif resp.status_code in [403, 429]:
                    if attempt < max_attempts - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                        time.sleep(sleep_time)
                        continue
                    else:
                        resp.raise_for_status()
                else:
                    resp.raise_for_status()
                    return resp  # 兜底返回，虽然常规下 raise_for_status 会中断
                    
            except Exception as e:
                # [修复] 对连接错误和超时执行外层重试(带抖动)，HTTPAdapter的Retry仅处理500级
                if attempt < max_attempts - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    continue
                raise e
                
        return None

    def get(self, url, timeout=8, encoding=None):
        """标准 GET 请求 (重试护航 + 连接池复用)"""
        try:
            # [恢复] 重新启用 self.session，享受自动重试和连接复用！
            # [升级] 替换为带抖动的防御性请求
            response = self._request_with_jitter('GET', url, timeout=timeout)
            if encoding and response is not None:
                response.encoding = encoding
            return response
        except Exception as e:
            from utils import HunterShield
            HunterShield.record(f"NetFail_GET | {url.split('?')[0]}", e)
            return None

    def get_fresh(self, url, params=None, timeout=6):
        """
        [修复] 真正的高频轮询通道 (打一枪换一个地方)
        强制声明 Connection: close，打破东财对 Keep-Alive 长连接的状态机追踪，防 TCP 掐断
        """
        try:
            # 1. 拷贝全局 headers，防止污染
            fresh_headers = self.headers.copy()
            # 2. [核心防杀] 强制关闭长连接，变成一次性抛弃型请求
            fresh_headers['Connection'] = 'close'
            
            # 3. 传入定制的 headers，_request_with_jitter 会自动补齐随机 User-Agent
            return self._request_with_jitter('GET', url, params=params, headers=fresh_headers, timeout=timeout)
        except Exception as e: 
            from utils import HunterShield
            HunterShield.record(f"NetFail_GET_FRESH | {url.split('?')[0]}", e)
            return None

    def post(self, url, json_data, headers, timeout=60):
        try:
            h = self.headers.copy()
            # [修复] 增加 or {} 兜底，防止外部传入 headers=None 导致 TypeError
            h.update(headers or {})
            # [核心修复] 强制大模型接口走抖动重试管道
            return self._request_with_jitter('POST', url, json=json_data, headers=h, timeout=timeout)
        except Exception as e:
            from utils import HunterShield
            HunterShield.record(f"NetFail_POST | {url.split('?')[0]}", e)
            return None


