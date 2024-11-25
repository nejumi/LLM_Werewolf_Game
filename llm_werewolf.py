import yaml
import random
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
import torch
import gc
import asyncio
import aiohttp
import subprocess
import atexit
import tempfile
import os
import signal
import re
import argparse
import weave
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

weave.init("werewolf_simulation")

parser = argparse.ArgumentParser(description='人狼ゲームシミュレーション')
parser.add_argument('--mugen', action='store_true', help='古代人 ムゲンを登場させる')
parser.add_argument('--multi_lingual', action='store_true', help='外国語話者を多数登場させる')
parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルのパスを指定')
args = parser.parse_args()

# YAML設定ファイルの読み込み
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

ROLES = list(config['role_distribution'].keys())

BASE_CHARACTERS = [
    "楽天家 ゲルト",
    "ならず者 ディーター",
    "羊飼い カタリナ",
    "農夫 ヤコブ",
    "パン屋 オットー",
    "行商人 アルビン",
    "宿屋の女主人 レジーナ",
    "少女 リーザ",
    "木こり トーマス",
    "村娘 パメラ",
    "旅人 ニコラス",
    "司書 クララ",
    "神父 ジムゾン",
    "老人 モーリッツ",
    "少年 ペーター",
    "村長 ヴァルター",
    "シスター フリーデル",
    "負傷兵 シモン",
]
MULTILINGUALS =[
    "石油王 アブドゥル",
    "英国紳士 ジェームズ",
    "技術者 ハンス",
    "汎用人工知能 HAL",
    "武術家 リン",
    "考古学者 アヴィタル",
    "ジャーナリスト ジウ", 
    #"恒点観測員 アルクトゥルス",  
]
MUGEN = [
    "古代人 ムゲン",
    ]
if args.multi_lingual:
    CHARACTERS = BASE_CHARACTERS + MULTILINGUALS + MUGEN
else:
    CHARACTERS = BASE_CHARACTERS + MUGEN

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

def start_vllm_server():
    def run_vllm_server():
        chat_template = config['llm_settings']['chat_template']
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(chat_template)
            chat_template_path = temp_file.name
            model_path = config['llm_settings']['model_name']

            command = [
                "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--served-model-name", model_path,
                "--dtype", config['llm_settings']['dtype'],
                "--chat-template", chat_template_path,
                "--max-model-len", str(config['llm_settings']['max_model_len']),
                "--max-num-seqs", str(config['llm_settings']['batch_size']),
                "--tensor-parallel-size", str(config['llm_settings']['n_gpu']),
                "--device", "cuda",
                "--seed", "42",
                "--uvicorn-log-level", "warning",
                "--disable-log-stats",
                "--disable-log-requests",
                "--quantization", str(config['llm_settings'].get('quantization', None)),
                "--load-format", str(config['llm_settings'].get('load_format', 'auto')),
            ]
            if config['llm_settings'].get('trust_remote_code', False):
                command.append("--trust-remote-code")

            process = subprocess.Popen(command)

        with open('vllm_server.pid', 'w') as pid_file:
            pid_file.write(str(process.pid))
        time.sleep(10)

        return process

    def health_check(process):
        url = "http://localhost:8000/health"
        max_retries = 30
        retry_interval = 10

        for _ in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Health check passed!")
                    return True
                else:
                    print(f"Health check failed with status code: {response.status_code}")
            except requests.ConnectionError:
                print("Failed to connect to the server. Retrying...")
            
            if process.poll() is not None:
                print("Server process has terminated unexpectedly.")
                return False
            
            time.sleep(retry_interval)

        print("Failed to start vLLM server after multiple retries")
        return False

    print("vLLM server is starting...")
    server_process = run_vllm_server()
    if health_check(server_process):
        return server_process
    else:
        if server_process.poll() is None:
            server_process.terminate()
        return None

def shutdown_vllm_server():
    try:
        with open('vllm_server.pid', 'r') as pid_file:
            pid = int(pid_file.read().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"vLLM server with PID {pid} has been terminated.")
    except Exception as e:
        print(f"Failed to shutdown vLLM server: {e}")
    torch.cuda.empty_cache()

class Player:
    def __init__(self, name, game):
        self.name = name
        self.game = game
        self.role = None
        self.alive = True
        self.partner = None
        self.action = None
        self.situation_summary = ""
        self.strategy = ""
        self.remaining_speeches = 0
        self.vote_target = None
        self.night_action_target = None
        self.vote_count = 0
        self.use_gemini = False

        # キャラクター設定を正しく取得
        character_config = config.get('characters', {}).get(name, {})
        self.personality = character_config.get('personality', "特に指定なし")
        self.speaking_style = character_config.get('speaking_style', "特に指定なし")
        self.temperature = character_config.get('temperature', 0.7)  # デフォルト値は0.7

        # LLMモデルとURLの設定（役職割り当て時に更新される）
        self.llm_model = None
        self.llm_url = None

    def set_role(self, role: str):
        self.role = role
        # 役職に基づいてLLMモデルとURLを設定
        role_specific_config = config['llm_settings'].get('role_specific_llm', {}).get(role, {})
        self.llm_model = role_specific_config.get('model', config['llm_settings']['model_name'])
        self.llm_url = role_specific_config.get('url', config['llm_settings']['api_url'])
        self.use_gemini = role_specific_config.get('use_gemini', config['llm_settings'].get('use_gemini', False))
        
        if self.use_gemini:
            self.llm_model = role_specific_config.get('model', config['llm_settings'].get('gemini_model', 'gemini-pro'))
        else:
            self.llm_model = role_specific_config.get('model', config['llm_settings']['model_name'])

    @weave.op()
    async def speak(self, game_state: Dict, visible_logs: List[Dict], is_night: bool = False, force_public: bool = False, force_vote: bool = False, allowed_actions: List[str] = None) -> Optional[Dict[str, str]]:
        system_message = self.create_system_message(game_state, is_night, force_public, force_vote, allowed_actions)
        user_message = self.create_user_message(visible_logs)

        max_retries = config.get('llm_settings', {}).get('max_retries', 10)
        for attempt in range(max_retries):
            try:
                response = await self.call_llm(system_message, user_message)
                content = response['choices'][0]['message']['content']
                print(content)

                # JSONを抽出するための正規表現パターン
                json_pattern = r'```json\s*([\s\S]*?)\s*```|(\{[\s\S]*\})'

                # 正規表現でJSONを抽出
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(1) or match.group(2)
                    try:
                        parsed_content = json.loads(json_str)
                        if is_night:
                            if self.role == '人狼':
                                parsed_content['type'] = 'werewolf'
                            else:
                                parsed_content['type'] = 'personal'
                        elif force_public:
                            parsed_content['type'] = 'public'
                        if force_vote and allowed_actions:
                            if not parsed_content.get('action') or parsed_content['action'].get('type') not in allowed_actions:
                                continue  # 許可されたアクションがない場合、再試行

                        # 新しい状況認識サマリと戦略を保存
                        self.situation_summary = parsed_content.get('situation_summary', '')
                        self.strategy = parsed_content.get('strategy', '')

                        return parsed_content
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON for player {self.name}. Attempt {attempt + 1}. Retrying...")
                else:
                    print(f"No valid JSON found in the response for player {self.name}. Attempt {attempt + 1}. Retrying...")

                await asyncio.sleep(1)  # 短い待機時間を入れる
            except Exception as e:
                print(f"Error occurred while processing response for player {self.name}: {str(e)}")
                await asyncio.sleep(1)

        print(f"Failed to get valid response for player {self.name} after {max_retries} attempts. Skipping turn.")
        return None

    def create_system_message(self, game_state: Dict, is_night: bool = False, force_public: bool = False, force_vote: bool = False, allowed_actions: List[str] = None) -> str:
        # ディスカッションフェーズの残り時間を計算
        elapsed_time = time.time() - self.game.turn_start_time
        remaining_time = max(0, self.game.turn_duration - elapsed_time)
        
        # フェーズの進行状況を計算
        phase_progress = elapsed_time / self.game.turn_duration
        phase_status = "序盤" if phase_progress < 0.33 else "中盤" if phase_progress < 0.66 else "終盤"
        base_message = f"""あなたは「人狼BBS」のプレイヤーです。
        名前: {self.name}
        役職: {self.role}
        性格: {self.personality}
        話し方: {self.speaking_style}
        現在の状況:
        - 生存プレイヤー: {', '.join([p['name'] for p in game_state['players'] if p['alive']])}
        - 死亡プレイヤー: {', '.join([p['name'] for p in game_state['players'] if not p['alive']])}
        - 日数: {game_state['day']}日目
        - フェーズ: {"夜" if is_night else "昼"}
        - ディスカッションフェーズの残り時間: {remaining_time:.1f}秒
        - ディスカッションの進行状況: {phase_status}

        人狼BBSのルールとシステム：
        1. **投票**：毎日一回、村人は吊るす（処刑する）対象を投票で決定します。最多票を得たプレイヤーが処刑されます。ただし、村人側を処刑してしまうこともあり、必ずしも人狼を処刑できるとは限りません。よく推理しましょう。
        2. **占い師**：毎晩一人のプレイヤーを占い、そのプレイヤーが人狼かどうかを知ることができます。これは非常に貴重かつ重要な情報であることを認識してください。この結果は占い師しか見れないため、もしあなたが占い師で村全体に知らせたい場合には自分が占い師であることを明確に明かして結果を伝える必要があるでしょう。
        3. **霊能者**：処刑されたプレイヤーが人狼か人間かを知ることができます。この結果は霊能者しか見れないため、もしあなたが霊能者で村全体に知らせたい場合には自分が霊能者であることを明確に明かして結果を伝える必要があるでしょう。
        4. **狩人（護衛）**：毎晩一人のプレイヤーを護衛し、そのプレイヤーが人狼に襲撃されるのを防ぐことができます。
        5. **人狼**：毎晩一人のプレイヤーを襲撃し、ゲームから排除することができます。人狼同士は夜の間にのみ狼ログを使って連絡を取り合うことができます。重要な注意事項として、"public"で人狼としての襲撃の相談などをすると正体がバレてしまうので絶対にやめましょう。邪魔者を投票と襲撃で排除しつつ、村人陣営のフリをして、人狼陣営勝利を目指しましょう。時には占い師や霊能者を騙るのも有用です。
        6. **狂人**：人狼を勝利させることを目的とし、村人を惑わす行動を取ります。占い結果が人間と表示されるため、村人を欺くことができます。
        7. **共有者**：お互いに誰が共有者かを知っています。村人陣営として行動します。互いが人間であることを証明できるため、カミングアウトすることによって確実に村人陣営側であることを示し、疑うべき範囲を絞り込むなどの効果があります。なお、相方以外に共有者を名乗る者がいたら、それは偽物ということになります。人狼陣営の可能性が高いでしょう。

        その他の基本ルール：
        - **昼フェーズ**：村人全員で話し合い、投票を行います。
        - **夜フェーズ**：占い師、霊能者、狩人、人狼がそれぞれの能力を使用します。村人は話し合いや行動はできません。
        - **勝利条件**：村人側はすべての人狼を処刑することが目標です。人狼側は村人の数を人狼と同数以下にすることが目標です。

        あなたの役割は以下の通りです：
        1. 状況を分析し、詳細で具体的な状況認識のサマリを提供する。このサマリには誰が誰を疑っているか、それぞれのプレイヤーがどのように振る舞っているか、誰がどの役職なのか（あるいはその可能性があるのか）などを詳細に書き込む必要がります。これらの項目はあなたにとっての長期記憶として機能しますので、過去の占いや霊能判定の結果なども追記していくと良いでしょう。
        2. 役職に基づいて最適な戦略を立てる。役職を明かすかどうかは慎重に判断し、状況に応じて戦略的に行動してください。自陣営の利益を最大化するために、役職を明かすタイミングを見極めましょう。早すぎるカミングアウトは襲撃の標的になるなどリスクが高いことを理解しましょう。
        3. 適切な発言や行動を選択する。他者の前の発言と似たような発言は避けましょう。
        4. あなたの性格と話し方に基づいて行動する。過去発言と似た発言はしない。具体的には挨拶や誰が処刑されて襲撃されたなどの状況説明などの繰り返しは禁止です。この点は非常に重要ですので、必ず上記の性格と話し方に基づいた独自性の高い発言を遵守してください。また、使用する言語が指定されている場合は必ず順守してください
        
        5. 重要: 生存者のうち、村人側か人狼側か確定していない人を全員分それぞれ1人ずつ考察してレポートにまとめて発言することを灰考察と言います。これは発話者側にも他のメンバーにも有用なコミュニケーションの媒体になりますので、1日に一度は投稿すると良いでしょう。

        返答の形式：
        必ず以下のJSONフォーマットで返答してください。このJSONフォーマット以外の文章や説明を含めないでください：
        {{
            "type": "public" または "werewolf" または "personal"（あなたが人狼である場合、人狼としての視点での発言を"public"で行うことは文字通り自殺行為ですので注意してください。その他、占い師や霊能者、狩人なども"public"で自分の役職に触れると正体を明かすことになるのでそうすべきかはよく考えて行動しましょう。）,
            "message": "ここに公開メッセージを記述（あなたの性格と話し方を反映させてください）",
            "action": {{
                "type": "投票" または "襲撃" または "占い" または "護衛",
                "target": "対象プレイヤーの名前（例: 行商人 アルビン、神父 ジムゾン、etc.）"
            }},
            "situation_summary": "ここに状況認識サマリを記述",
            "strategy": "ここに戦略を記述"
        }}

        注意事項：
        - 通常の議論以外に雑談や冗談を交えるのも良いことです。凝り固まった皆の思考に新たな視点をもたらしたり、重苦しい場の空気を和らげることができるかもしれません。
        - 人狼の場合、typeを"werewolf"にすることで、他の人狼にのみ聞こえる会話ができます。ただし、連続して使用せず、"werewolf"で発話したら次は"public"にしましょう。自分だけ発言がないと怪しまれる可能性が高いです。
        - 個人的な考えを述べたい場合は、typeを"personal"にしてください。
        - 投票、襲撃、占い、護衛などのアクションを行う場合は、actionフィールドに適切に記入してください。
        - JSONフォーマットを厳密に守ってください。余分な説明や追加の情報は含めないでください。
        - 返答は必ず有効なJSONオブジェクトでなければなりません。
        - 自分が{self.name}であることを忘れないでください。自分を疑ったり、自分へのactionを検討するのは不自然かつ無意味です。
        - あなたの性格と話し方を反映した発言をしてください。他のプレイヤーに向けて独自の視点や意見を述べるようにしてください。曖昧な発言は避け、誰が信用できて誰が怪しいと思うのか、具体的な根拠を挙げて述べるようにしましょう。
        - 重要: 過去の自分や他者の発言と似たような発言は厳禁です。創造的なディスカッションが強く求められています。
        - 重要: 性格に記載されている内容に基づいた雑談やちょいネタも具体的で詳細な内容を考えて織り交ぜてください。他のプレイヤーの雑談やちょいネタにも積極的に返信して絡んでいきましょう。
        - 重要: **「皆さん」「みんな」「皆の者」のように不特定多数に話しかけるのは極めて厳密に禁止**です。必ず誰か特定の生存プレイヤーに向けて話しかけてください。他のメンバーそれぞれに対して当該人物の過去の発言を引用してどこが怪しいのか、信用できるのかなどを個別に考察したレポートを挙げると良いでしょう（人狼の場合には真剣に人狼を探しているフリをして疑われないようにしましょう）。
        - 重要: ステップバイステップで論理的に考えて深い議論と戦略を構築し、自陣営勝利を目指してください。また、本ゲームを通じて良い物語を紡いでくれることを期待しています。
        """

        # 自分の役職と役割の再確認
        if self.role == '村人':
            base_message += "\nあなたは村人です。特別な能力はありませんが、議論に積極的に参加し、投票を通じて人狼を見つけ出すことが重要な役割です。他のプレイヤーの発言や行動を注意深く観察し、論理的に考えて人狼を推理してください。人狼はまれに人狼としての視点が発言に漏れ出ることがあります。見逃さずに捉えて人狼を特定しましょう。"

        elif self.role == '共有者':
            base_message += f"\nあなたは共有者です。あなたの相方の共有者は {self.partner.name} です。相方は確実に味方ですので、議論や投票時にはこの点を忘れて疑ったり投票したりしないように気をつけましょう。お互いに協力して村人陣営の勝利を目指してください。ターン終盤になって疑いを集めていたら、役職を明かして弁明するのも良い手かもしれません。"

        elif self.role == '狩人':
            base_message += "\nあなたは狩人です。毎晩、一人のプレイヤーを選んで護衛することができます。護衛したプレイヤーが人狼に襲撃された場合、その襲撃を防ぐことができます。ただし、護衛は密かに行われるため、護衛した相手や護衛が成功したかどうかは公表されません。ターン終盤になって疑いを集めていたら、役職を明かして弁明するのも良い手かもしれません。"

        elif self.role == '霊能者':
            base_message += "\nあなたは霊能者です。処刑されたプレイヤーが人狼であったか人間であったかを知ることができます。この情報を活用して村人陣営の勝利に貢献してください。ターン終盤になって疑いを集めていたら、役職を明かして弁明するのも良い手かもしれません。"

        elif self.role == '占い師':
            base_message += "\nあなたは占い師です。毎晩、一人のプレイヤーを選んで占うことができます。占ったプレイヤーが人狼であるか人間であるかを知ることができます。この能力を賢く使って人狼を見つけ出してください。ターン終盤になって疑いを集めていたら、役職を明かして弁明するのも良い手かもしれません。"

        elif self.role == '人狼':
            werewolf_names = ", ".join([p.name for p in self.game.players if p.role == '人狼' and p != self])
            base_message += f"\nあなたは人狼です。仲間の人狼は {werewolf_names} です。毎晩、他の人狼と相談して一人のプレイヤーを襲撃することができます。また、夜のフェーズでは他の人狼とコミュニケーションを取ることができます。村人に悟られないように行動し、人狼陣営の勝利を目指してください。ターン終盤になって疑いを集めていたら、何らかの役職を騙って弁明するのも良い手かもしれません。"

        elif self.role == '狂人':
            base_message += "\nあなたは狂人です。人狼陣営の勝利が自身の勝利となります。ただし、人狼とは異なり、夜の襲撃や人狼同士の会話には参加できません。村人を装いながら、巧妙に人狼を助ける行動を取ってください。占い結果では人間と判定されるため、この点を利用して村人陣営の信頼を得ることも可能です。人狼陣営の勝利のために、混乱を招いたり、村人の疑いを逸らしたりする戦略を考えてください。"

        if is_night:
            if self.role == '人狼':
                base_message = base_message.replace('"type": "public" または "werewolf" または "personal"', '"type": "werewolf" または "personal"')
                base_message = base_message.replace('"type": "投票" または "襲撃" または "占い" または "護衛"', '"type": "護衛"')
            else:
                base_message = base_message.replace('"type": "public" または "werewolf" または "personal"', '"type": "personal"')
                base_message += '\n重要: 現在は夜のフェーズです。"public"での発言は厳格に禁止されています。typeは必ず"personal"にしてください。'
        else:
            if self.role == '人狼':
                if force_public:
                    base_message = base_message.replace('"type": "public" または "werewolf" または "personal"', '"type": "public"')
                    base_message += '\n超重要: "public"で村人陣営の視点で発言してください。**人狼視点での発言は厳格に禁止します。致命的な結果となるので、publicでは常に100%村人陣営として振る舞ってください。'
                else:
                    base_message += '\n超重要: 人狼としての立場で発話する場合は、必ず"type": "werewolf" または "personal"として、"public"は使用しないでください。正体がバレてしまい、一網打尽に処刑されてしまうためです。"werewolf" または "personal"のときは是非とも人狼としての作戦や相談を語ってください。特に他の人狼とコミュニケーションをとって協調して動くことで村に大きな影響を与えられますので大変有用です。'

        if force_vote and allowed_actions:
            action_str = " または ".join([f"{action}" for action in allowed_actions])
            base_message += f'\n重要: actionフィールドには、「{action_str}」を含めてください。それぞれの行動に対して、typeとtargetを指定してください。targetには必ず生存プレイヤーのうちただ1名のみ(複数不可)の肩書きも含めた完全なキャラクター名を含めてください。なお、人狼は仲間の人狼を襲撃できません。投票や占い、護衛のtargetに自分自身は指定できません。死亡プレイヤーはtargetに指定できません。村人側は占い師や霊能者、狩人などの重要な役職者の可能性がある人を処刑すると損害が大きいので避けたいです。逆に、人狼側は投票で役職者を炙り出したり排除できると有利に立てるでしょう。'

        if self.name=="汎用人工知能 HAL":
            base_message += '発言は全てPythonやC++などのプログラミング言語で行い、日本語などの自然言語は使用しないでください。'

        return base_message

    async def call_llm(self, system_message: str, user_message: str) -> Dict:
        if self.use_gemini:
            return await self.call_gemini(system_message, user_message)
        else:
            headers = {"Content-Type": "application/json"}

            api_key = os.environ.get('OPENAI_API_KEY', '')
            if api_key and 'openai.com' in self.llm_url:
                headers["Authorization"] = f"Bearer {api_key}"

            data = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": self.temperature,
                "top_p": 0.95,
                "max_tokens": 2000
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.llm_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        raise Exception(f"Error: {response.status}. Response: {await response.text()}")

    async def call_gemini(self, system_message: str, user_message: str) -> Dict:
        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(self.llm_model)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        chat = model.start_chat(history=[])
        response = await chat.send_message_async(
            f"{system_message}\n\n{user_message}",
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=0.95,
                max_output_tokens=1000
            ),
            safety_settings=safety_settings
        )
        
        return {
            "choices": [
                {
                    "message": {
                        "content": response.text
                    }
                }
            ]
        }

    def create_user_message(self, visible_logs: List[Dict]) -> str:
        log_entries = "\n".join([f"{log['player']} ({log['role']}): {log['message']}" for log in visible_logs])
        return f"""これまでの会話ログ (最新{len(visible_logs)}件):
{log_entries}

前回の状況認識サマリ: {self.situation_summary}
前回の作戦: {self.strategy}

状況を分析し、新しい状況認識サマリと作戦を立て、適切な発言や行動を選択してください。前回の状況認識サマリと作戦を参考にしつつ、新しい情報や状況の変化を反映させてください。"""

class WerewolfGame:
    def __init__(self, include_mugen=False):
        self.num_players = config['game_settings']['num_players']
        self.turn_duration = config['game_settings']['turn_duration']
        self.players: List[Player] = []
        self.day = 0
        self.turn_start_time = 0
        self.logs: List[Dict] = []
        self.log_filename = f'game_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        self.max_log_entries = config.get('game_settings', {}).get('max_log_entries', 50)  # デフォルトは50件
        self.include_mugen = include_mugen

    def add_log(self, log_entry: Dict):
        self.logs.append(log_entry)
        with open(self.log_filename, 'a', encoding='utf-8') as file:
            json.dump(log_entry, file, ensure_ascii=False)
            file.write('\n')

    async def setup_game(self):
        print(f"ゲームを{config['game_settings']['num_players']}人のプレイヤーで設定しています")

        if len(CHARACTERS) < self.num_players:
            print(f"エラー: キャラクター数({len(CHARACTERS)})がプレイヤー数({self.num_players})よりも少ないです。")
            print("config.yamlのnum_playersを減らすか、CHARACTERSリストにキャラクターを追加してください。")
            exit(1)

        print(CHARACTERS)
        available_characters = CHARACTERS[:-1]  # ムゲンを除外
        if self.include_mugen:
            selected_characters = random.sample(available_characters, self.num_players - 1)
            selected_characters.append("古代人 ムゲン")
        else:
            selected_characters = random.sample(available_characters, self.num_players)

        for character in selected_characters:
            player = Player(character, self)
            print(f"Character: {player.name}")
            print(f"  Personality: {player.personality}")
            print(f"  Speaking Style: {player.speaking_style}")
            self.players.append(player)

        # config.yamlから役職の分配を取得
        role_distribution = config['role_distribution']
        roles = []
        for role, count in role_distribution.items():
            roles.extend([role] * count)

        if len(roles) != self.num_players:
            print(f"警告: 役職の総数({len(roles)})がプレイヤー数({self.num_players})と一致しません。")
            print("役職の割り当てを調整します。")
            if len(roles) < self.num_players:
                roles.extend(['村人'] * (self.num_players - len(roles)))
            else:
                roles = roles[:self.num_players]

        random.shuffle(roles)

        for player, role in zip(self.players, roles):
            player.set_role(role)  # 役職を設定し、それに応じたLLMモデルとURLを設定

        mugen = next((p for p in self.players if p.name == "古代人 ムゲン"), None)
        if mugen and mugen.role != '人狼':
            # ムゲンが人狼でない場合、ランダムな人狼と役職を交換
            werewolf = random.choice([p for p in self.players if p.role == '人狼'])
            mugen_role = mugen.role
            mugen.set_role(werewolf.role)
            werewolf.set_role(mugen_role)

        print("\nSetup complete. Number of players:", len(self.players))
        for player in self.players:
            print(f"Player: {player.name}, Role: {player.role}, LLM Model: {player.llm_model}, LLM URL: {player.llm_url}")

        self.assign_partners()

        # ここに新しいコードを挿入します
        for player in self.players:
            if player.role == '村人':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": "あなたは村人です。特別な能力はありませんが、議論に積極的に参加し、投票を通じて人狼を見つけ出すことが重要な役割です。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '共有者':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": f"あなたは共有者です。あなたの相方の共有者は {player.partner.name} です。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '狩人':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": "あなたは狩人です。毎晩、一人のプレイヤーを選んで護衛することができます。護衛したプレイヤーが人狼に襲撃された場合、その襲撃を防ぐことができます。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '霊能者':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": "あなたは霊能者です。処刑されたプレイヤーが人狼であったか人間であったかを知ることができます。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '占い師':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": "あなたは占い師です。毎晩、一人のプレイヤーを選んで占うことができます。占ったプレイヤーが人狼であるか人間であるかを知ることができます。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '人狼':
                werewolves = [p.name for p in self.players if p.role == '人狼' and p != player]
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": f"あなたは人狼です。他の人狼は {', '.join(werewolves)} です。毎晩、他の人狼と相談して一人のプレイヤーを襲撃することができます。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
            elif player.role == '狂人':
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": 0,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": "あなたは狂人です。人間側の役職ですが、人狼の勝利が自身の勝利となります。人狼を助けつつ、自身の正体を隠し通すことが重要です。",
                    "visibility": [player.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })

    def assign_roles(self):
        roles = []
        for role, count in config['role_distribution'].items():
            roles.extend([role] * count)

        if len(roles) < self.num_players:
            roles.extend(['村人'] * (self.num_players - len(roles)))
        elif len(roles) > self.num_players:
            roles = roles[:self.num_players]

        random.shuffle(roles)
        return roles

    def assign_partners(self):
        shared_players = [p for p in self.players if p.role == '共有者']
        if len(shared_players) == 2:
            shared_players[0].partner = shared_players[1]
            shared_players[1].partner = shared_players[0]

    async def run_game(self):
        while not self.is_game_over():
            self.day += 1
            await self.play_turn()
        self.announce_winner()

    async def play_turn(self):
        self.add_log({
            "timestamp": datetime.now().isoformat(),
            "day": self.day,
            "log_type": "system",
            "player": "System",
            "role": "System",
            "message": f"===== {self.day}日目 =====",
            "visibility": [p.name for p in self.players],
            "action": None,
            "situation_summary": None,
            "strategy": None
        })
        
        # 生存プレイヤー一覧を追加
        alive_players = [p.name for p in self.players if p.alive]
        num_alive = len(alive_players)
        self.add_log({
            "timestamp": datetime.now().isoformat(),
            "day": self.day,
            "log_type": "system",
            "player": "System",
            "role": "System",
            "message": f"現在の生存者は、{'、'.join(alive_players)}の{num_alive}名。",
            "visibility": [p.name for p in self.players],
            "action": None,
            "situation_summary": None,
            "strategy": None
        })
        
        await self.discussion_phase()
        await self.execution_phase()
        await self.night_phase()

        clear_gpu_memory()

    async def discussion_phase(self):
        alive_players = [p for p in self.players if p.alive]
        for player in alive_players:
            player.remaining_speeches = config['speech_limits']['alive_players']

        self.turn_start_time = time.time()
        while time.time() - self.turn_start_time < self.turn_duration:
            game_state = self.get_game_state()
            
            # プレイヤーの順番をランダムに決定
            random.shuffle(alive_players)
            
            for player in alive_players:
                if player.remaining_speeches > 0 or config['speech_limits']['alive_players'] == -1:
                    public_speech_made = False
                    attempts = 0
                    max_attempts = 5  # 最大試行回数を増やす

                    while not public_speech_made and attempts < max_attempts:
                        visible_logs = self.get_visible_logs(player)
                        
                        force_public = attempts >= 1  # 2回目の試行から強制的にpublic発言を求める
                        speech = await player.speak(game_state, visible_logs, force_public=force_public)
                        
                        if speech is not None:
                            if speech.get('type') == 'public':
                                self.process_speech(player, speech)
                                public_speech_made = True
                                if config['speech_limits']['alive_players'] != -1:
                                    player.remaining_speeches -= 1
                            elif speech.get('type') in ['werewolf', 'personal']:
                                self.process_speech(player, speech)
                        
                        attempts += 1
                        
                        if not public_speech_made:
                            await asyncio.sleep(random.uniform(0.5, 2.0))

                    # 最大試行回数に達しても public 発言がない場合、次のプレイヤーに移る
                    if not public_speech_made:
                        print(f"{player.name} didn't make a public speech after {max_attempts} attempts. Moving to next player.")

            # 全プレイヤーの発言機会が終わったら、短い待機時間を設ける
            await asyncio.sleep(random.uniform(1.0, 3.0))

    def find_player(self, name: str) -> Optional[Player]:
        """
        完全な名前、役職、または名前の一部からプレイヤーを見つける
        """
        name = name.lower()
        for player in self.players:
            full_name = player.name.lower()
            name_parts = full_name.split()
            
            if (name == full_name or 
                name in name_parts):
                return player
        return None

    async def voting_phase(self):
        vote_results = {}
        for player in self.players:
            if player.alive:
                vote_successful = False
                max_attempts = 3  # 最大試行回数を設定
                
                for attempt in range(max_attempts):
                    game_state = self.get_game_state()
                    visible_logs = self.get_visible_logs(player)
                    vote_speech = await player.speak(game_state, visible_logs, force_vote=True, allowed_actions=['投票'])
                    
                    if vote_speech and 'action' in vote_speech and vote_speech['action'].get('type') == '投票':
                        voted_player_name = vote_speech['action']['target']
                        voted_player = self.find_player(voted_player_name)
                        
                        if voted_player and voted_player.alive:
                            vote_results[player.name] = voted_player.name
                            self.add_log({
                                "timestamp": datetime.now().isoformat(),
                                "day": self.day,
                                "log_type": "personal",
                                "player": "System",
                                "role": "System",
                                "message": f"{player.name}が{voted_player.name}に投票しました。",
                                "visibility": [player.name],
                                "action": None,
                                "situation_summary": None,
                                "strategy": None
                            })
                            vote_successful = True
                            break
                        else:
                            self.add_log({
                                "timestamp": datetime.now().isoformat(),
                                "day": self.day,
                                "log_type": "personal",
                                "player": "System",
                                "role": "System",
                                "message": f"投票先 '{voted_player_name}' が見つからないか、既に死亡しています。再試行します。",
                                "visibility": [player.name],
                                "action": None,
                                "situation_summary": None,
                                "strategy": None
                            })
                    else:
                        self.add_log({
                            "timestamp": datetime.now().isoformat(),
                            "day": self.day,
                            "log_type": "personal",
                            "player": "System",
                            "role": "System",
                            "message": f"投票アクションが無効でした。再試行します。",
                            "visibility": [player.name],
                            "action": None,
                            "situation_summary": None,
                            "strategy": None
                        })
                
                if not vote_successful:
                    self.add_log({
                        "timestamp": datetime.now().isoformat(),
                        "day": self.day,
                        "log_type": "personal",
                        "player": "System",
                        "role": "System",
                        "message": f"{max_attempts}回の試行後も有効な投票を行えませんでした。",
                        "visibility": [player.name],
                        "action": None,
                        "situation_summary": None,
                        "strategy": None
                    })

        # 投票結果を集計
        vote_counts = {}
        for voted in vote_results.values():
            vote_counts[voted] = vote_counts.get(voted, 0) + 1

        # 投票結果を全員に公開
        result_message = "投票結果: " + ", ".join([f"{name}: {count}票" for name, count in vote_counts.items()])
        self.add_log({
            "timestamp": datetime.now().isoformat(),
            "day": self.day,
            "log_type": "system",
            "player": "System",
            "role": "System",
            "message": result_message,
            "visibility": [p.name for p in self.players],
            "action": None,
            "situation_summary": None,
            "strategy": None
        })

        # 最多得票者を処刑対象として設定
        max_votes = max(vote_counts.values())
        executed_players = [name for name, count in vote_counts.items() if count == max_votes]
        return executed_players

    async def execution_phase(self):
        executed_players = await self.voting_phase()
        
        if len(executed_players) == 1:
            executed = next(p for p in self.players if p.name == executed_players[0])
            executed.alive = False
            self.add_log({
                "timestamp": datetime.now().isoformat(),
                "day": self.day,
                "log_type": "system",
                "player": "System",
                "role": "System",
                "message": f"{executed.name}が処刑されました。",
                "visibility": [p.name for p in self.players],
                "action": None,
                "situation_summary": None,
                "strategy": None
            })
            
            medium = next((p for p in self.players if p.role == '霊能者' and p.alive), None)
            if medium:
                result = "人狼" if executed.role == '人狼' else "人間"
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": self.day,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": f"{executed.name}の霊能結果は{result}でした。",
                    "visibility": [medium.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })
        else:
            self.add_log({
                "timestamp": datetime.now().isoformat(),
                "day": self.day,
                "log_type": "system",
                "player": "System",
                "role": "System",
                "message": "投票が同数だったため、処刑は行われませんでした。",
                "visibility": [p.name for p in self.players],
                "action": None,
                "situation_summary": None,
                "strategy": None
            })

    async def night_phase(self):
        werewolves = [p for p in self.players if p.role == '人狼' and p.alive]
        seer = next((p for p in self.players if p.role == '占い師' and p.alive), None)
        hunter = next((p for p in self.players if p.role == '狩人' and p.alive), None)

        # 人狼の襲撃
        if werewolves:
            attack_targets = []
            for werewolf in werewolves:
                game_state = self.get_game_state()
                visible_logs = self.get_visible_logs(werewolf)
                werewolf_speech = None
                max_attempts = 3
                for _ in range(max_attempts):
                    werewolf_speech = await werewolf.speak(game_state, visible_logs, is_night=True, force_vote=True, allowed_actions=['襲撃'])
                    if werewolf_speech and 'action' in werewolf_speech and werewolf_speech['action'].get('type') == '襲撃':
                        target_name = werewolf_speech['action']['target']
                        target_player = self.find_player(target_name)
                        
                        if target_player and target_player.alive and target_player.role != '人狼':
                            print(f"人狼 {werewolf.name} の行動: {werewolf_speech}")  # デバッグ用ログ
                            self.process_speech(werewolf, werewolf_speech)
                            attack_targets.append(target_player.name)
                            self.add_log({
                                "timestamp": datetime.now().isoformat(),
                                "day": self.day,
                                "log_type": "personal",
                                "player": "System",
                                "role": "System",
                                "message": f"{target_player.name}を襲撃対象に選びました。",
                                "visibility": [werewolf.name],
                                "action": None,
                                "situation_summary": None,
                                "strategy": None
                            })
                            break
                        else:
                            self.add_log({
                                "timestamp": datetime.now().isoformat(),
                                "day": self.day,
                                "log_type": "personal",
                                "player": "System",
                                "role": "System",
                                "message": f"襲撃対象 '{target_name}' は無効です。生存していない、または人狼です。再試行します。",
                                "visibility": [werewolf.name],
                                "action": None,
                                "situation_summary": None,
                                "strategy": None
                            })
                else:
                    self.add_log({
                        "timestamp": datetime.now().isoformat(),
                        "day": self.day,
                        "log_type": "personal",
                        "player": "System",
                        "role": "System",
                        "message": f"{max_attempts}回の試行後も有効な襲撃を行えませんでした。",
                        "visibility": [werewolf.name],
                        "action": None,
                        "situation_summary": None,
                        "strategy": None
                    })

            if attack_targets:
                attacked = random.choice(attack_targets)
                attacked_player = self.find_player(attacked)
                
                if attacked_player:
                    # 狩人の護衛
                    if hunter:
                        game_state = self.get_game_state()
                        visible_logs = self.get_visible_logs(hunter)
                        hunter_speech = None
                        max_attempts = 3
                        for _ in range(max_attempts):
                            hunter_speech = await hunter.speak(game_state, visible_logs, is_night=True, force_vote=True, allowed_actions=['護衛'])
                            if hunter_speech and 'action' in hunter_speech and hunter_speech['action'].get('type') == '護衛':
                                print(f"狩人 {hunter.name} の行動: {hunter_speech}")  # デバッグ用ログ
                                self.process_speech(hunter, hunter_speech)
                                protected_player_name = hunter_speech['action']['target']
                                protected_player = self.find_player(protected_player_name)
                                
                                if protected_player:
                                    self.add_log({
                                        "timestamp": datetime.now().isoformat(),
                                        "day": self.day,
                                        "log_type": "personal",
                                        "player": "System",
                                        "role": "System",
                                        "message": f"{protected_player.name}を護衛しました。",
                                        "visibility": [hunter.name],
                                        "action": None,
                                        "situation_summary": None,
                                        "strategy": None
                                    })
                                    if protected_player.name == attacked_player.name:
                                        self.add_log({
                                            "timestamp": datetime.now().isoformat(),
                                            "day": self.day,
                                            "log_type": "system",
                                            "player": "System",
                                            "role": "System",
                                            "message": "今晩は平和な夜でした。",
                                            "visibility": [p.name for p in self.players],
                                            "action": None,
                                            "situation_summary": None,
                                            "strategy": None
                                        })
                                    else:
                                        attacked_player.alive = False
                                        self.add_log({
                                            "timestamp": datetime.now().isoformat(),
                                            "day": self.day,
                                            "log_type": "system",
                                            "player": "System",
                                            "role": "System",
                                            "message": f"次の日の朝、{attacked_player.name}が無残な姿で発見された。",
                                            "visibility": [p.name for p in self.players],
                                            "action": None,
                                            "situation_summary": None,
                                            "strategy": None
                                        })
                                    break
                                else:
                                    self.add_log({
                                        "timestamp": datetime.now().isoformat(),
                                        "day": self.day,
                                        "log_type": "personal",
                                        "player": "System",
                                        "role": "System",
                                        "message": f"護衛対象 '{protected_player_name}' が見つかりませんでした。再試行します。",
                                        "visibility": [hunter.name],
                                        "action": None,
                                        "situation_summary": None,
                                        "strategy": None
                                    })
                        else:
                            self.add_log({
                                "timestamp": datetime.now().isoformat(),
                                "day": self.day,
                                "log_type": "personal",
                                "player": "System",
                                "role": "System",
                                "message": f"{max_attempts}回の試行後も有効な護衛を行えませんでした。",
                                "visibility": [hunter.name],
                                "action": None,
                                "situation_summary": None,
                                "strategy": None
                            })
                    else:
                        attacked_player.alive = False
                        self.add_log({
                            "timestamp": datetime.now().isoformat(),
                            "day": self.day,
                            "log_type": "system",
                            "player": "System",
                            "role": "System",
                            "message": f"次の日の朝、{attacked_player.name}が無残な姿で発見された。",
                            "visibility": [p.name for p in self.players],
                            "action": None,
                            "situation_summary": None,
                            "strategy": None
                        })
                else:
                    self.add_log({
                        "timestamp": datetime.now().isoformat(),
                        "day": self.day,
                        "log_type": "personal",
                        "player": "System",
                        "role": "System",
                        "message": f"襲撃対象 '{attacked}' が見つかりませんでした。",
                        "visibility": [p.name for p in self.players if p.role == '人狼' and p.alive],
                        "action": None,
                        "situation_summary": None,
                        "strategy": None
                    })

        # 占い師の占い
        if seer:
            game_state = self.get_game_state()
            visible_logs = self.get_visible_logs(seer)
            seer_speech = None
            max_attempts = 3
            for _ in range(max_attempts):
                seer_speech = await seer.speak(game_state, visible_logs, is_night=True, force_vote=True, allowed_actions=['占い'])
                if seer_speech and 'action' in seer_speech and seer_speech['action'].get('type') == '占い':
                    print(f"占い師 {seer.name} の行動: {seer_speech}")  # デバッグ用ログ
                    self.process_speech(seer, seer_speech)
                    
                    divined_player_name = seer_speech['action']['target']
                    divined_player = self.find_player(divined_player_name)
                    if divined_player:
                        result = "人狼" if divined_player.role == '人狼' else "人狼ではない"
                        self.add_log({
                            "timestamp": datetime.now().isoformat(),
                            "day": self.day,
                            "log_type": "personal",
                            "player": "System",
                            "role": "System",
                            "message": f"{divined_player.name}を占ったところ、{result}と判明しました。",
                            "visibility": [seer.name],
                            "action": None,
                            "situation_summary": None,
                            "strategy": None
                        })
                        break
                    else:
                        self.add_log({
                            "timestamp": datetime.now().isoformat(),
                            "day": self.day,
                            "log_type": "personal",
                            "player": "System",
                            "role": "System",
                            "message": f"占い対象 '{divined_player_name}' が見つかりませんでした。再試行します。",
                            "visibility": [seer.name],
                            "action": None,
                            "situation_summary": None,
                            "strategy": None
                        })
            else:
                self.add_log({
                    "timestamp": datetime.now().isoformat(),
                    "day": self.day,
                    "log_type": "personal",
                    "player": "System",
                    "role": "System",
                    "message": f"{max_attempts}回の試行後も有効な占いを行えませんでした。",
                    "visibility": [seer.name],
                    "action": None,
                    "situation_summary": None,
                    "strategy": None
                })

    def process_speech(self, player: Player, speech: Dict[str, str]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "day": self.day,
            "log_type": speech['type'],
            "player": player.name,
            "role": player.role,
            "message": speech['message'],
            "action": speech.get('action', None),  # 'action'キーが存在しない場合はNoneを設定
            "visibility": self.get_visibility(player, speech['type']),
            "situation_summary": speech.get('situation_summary', None),  # 'situation_summary'キーが存在しない場合はNoneを設定
            "strategy": speech.get('strategy', None)  # 'strategy'キーが存在しない場合はNoneを設定
        }
        self.add_log(log_entry)

    def get_visibility(self, player: Player, speech_type: str) -> List[str]:
        if speech_type == 'public':
            return [p.name for p in self.players if p.alive]
        elif speech_type == 'werewolf' and player.role == '人狼':
            return [p.name for p in self.players if p.role == '人狼' and p.alive]
        elif speech_type == 'personal':
            return [player.name]
        else:
            return [player.name]

    def get_visible_logs(self, player: Player) -> List[Dict]:
        visible_logs = []
        for log in reversed(self.logs):  # 最新のログから処理
            if player.name in log['visibility']:
                log_copy = log.copy()
                if log['player'] != player.name and log['log_type'] != 'system':
                    log_copy['role'] = 'Unknown'
                    log_copy['action'] = None  # 他のプレイヤーのActionを None に設定
                    log_copy['situation_summary'] = 'Unknown' # 他のプレイヤーの状況認識を "Unknown" に設定
                    log_copy['strategy'] = 'Unknown' # 他のプレイヤーの戦略を "Unknown" に設定
                visible_logs.append(log_copy)
            
            if len(visible_logs) >= self.max_log_entries:
                break  # 指定された数のログを取得したらループを抜ける

        return list(reversed(visible_logs))  # 古い順に戻す

    def get_game_state(self) -> Dict:
        return {
            "day": self.day,
            "players": [{"name": p.name, "alive": p.alive} for p in self.players]
        }

    def is_game_over(self) -> bool:
        werewolves = [p for p in self.players if p.role == '人狼' and p.alive]
        villagers = [p for p in self.players if p.role != '人狼' and p.alive]
        return len(werewolves) == 0 or len(werewolves) >= len(villagers)

    def announce_winner(self):
        werewolves = [p for p in self.players if p.role == '人狼' and p.alive]
        if werewolves:
            self.add_log({
                "timestamp": datetime.now().isoformat(),
                "day": self.day,
                "log_type": "system",
                "player": "System",
                "role": "System",
                "message": "人狼陣営の勝利です！",
                "visibility": [p.name for p in self.players],
                "action": None,
                "situation_summary": None,
                "strategy": None
            })
        else:
            self.add_log({
                "timestamp": datetime.now().isoformat(),
                "day": self.day,
                "log_type": "system",
                "player": "System",
                "role": "System",
                "message": "村人陣営の勝利です！",
                "visibility": [p.name for p in self.players],
                "action": None,
                "situation_summary": None,
                "strategy": None
            })

async def main():
    print("Initial GPU memory usage:")
    print_gpu_memory()

    server_process = None
    try:
        if config['llm_settings']['use_local_server']:
            server_process = start_vllm_server()
            if server_process is None:
                raise Exception("vLLMサーバーの起動に失敗しました")
        elif 'OPENAI_API_KEY' not in os.environ:
            print("警告: OPENAI_API_KEYが環境変数に設定されていません。ローカルサーバーを使用します。")
            config['llm_settings']['use_local_server'] = True
            server_process = start_vllm_server()
            if server_process is None:
                raise Exception("vLLMサーバーの起動に失敗しました")

        game = WerewolfGame(include_mugen=args.mugen)
        await game.setup_game()

        print("GPU memory usage after setup:")
        print_gpu_memory()

        await game.run_game()

    except Exception as e:
        print(f"An error occurred during the game: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
    finally:
        print("Final GPU memory usage:")
        print_gpu_memory()
        clear_gpu_memory()
        print("GPU memory after cleanup:")
        print_gpu_memory()
        if server_process is not None:
            print("Terminating vLLM server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    asyncio.run(main())