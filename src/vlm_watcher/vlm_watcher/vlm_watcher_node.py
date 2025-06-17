#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subscribe /rgb_image at full frame-rate, sample 5 fps, send to GPT-4o,
publish /recovery_action when anomaly is detected.
Reads an external JSONL action list and asks the model to pick one.
"""
import argparse
import base64
import json
import threading
import time
from queue import Queue, Empty

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vla_interfaces.msg import Action
from dotenv import load_dotenv

#  GPT-4o client初期化
load_dotenv()
try:
    from openai import OpenAI
    _client = OpenAI()
except Exception:
    _client = None


def parse_args():
    p = argparse.ArgumentParser(description='Continuous VLM watcher')
    p.add_argument('--fps',         type=float, default=5.0,
                   help='Sampling FPS for VLM')
    p.add_argument('--prompt',      type=str,
                   default='Detect semantic anomaly and suggest action.',
                   help='Base prompt text')
    p.add_argument('--action-list', type=str, required=True,
                   help='Path to JSONL file of available actions')
    return p.parse_args()


class VLMWatcher(Node):
    def __init__(self, args):
        super().__init__('vlm_watcher')

        # サンプリング間隔
        self.min_interval = 1.0 / args.fps
        self.prompt       = args.prompt

        # JSONL アクションリスト読み込み
        self.actions = []
        with open(args.action_list, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    j = json.loads(line)
                    name = j.get('action')
                    tid  = j.get('target_id','')
                    if name:
                        self.actions.append({'action': name, 'target_id': tid})
                except json.JSONDecodeError:
                    self.get_logger().warning(f'Invalid JSONL line skipped: {line.strip()}')
        self.get_logger().info(f'Loaded {len(self.actions)} actions from JSONL')

        # Pub/Sub
        self.bridge  = CvBridge()
        self.sub_img = self.create_subscription(
            Image, '/rgb_image', self._cb_img, 10)
        self.pub_act = self.create_publisher(
            Action, '/recovery_action', 10)

        # 非同期ワーカー用キュー
        self.last_sent = 0.0
        self.q_req, self.q_res = Queue(), Queue()
        for _ in range(4):
            threading.Thread(target=self._worker, daemon=True).start()

        # 100 ms タイマーで結果処理
        self.create_timer(0.1, self._timer_res)

        self.get_logger().info('VLMWatcher ready')

    def _cb_img(self, msg: Image):
        now = time.time()
        if now - self.last_sent < self.min_interval:
            return
        self.last_sent = now

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        retval, buf = cv2.imencode('.jpg', cv_img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not retval:
            return
        self.q_req.put(buf.tobytes())

    def _timer_res(self):
        while not self.q_res.empty():
            content = self.q_res.get_nowait()
            try:
                j = json.loads(content)
                if j.get('anomaly','normal') != 'normal':
                    act = Action()
                    act.name      = j.get('action','')
                    act.target_id = j.get('target_id','')
                    self.pub_act.publish(act)
                    self.get_logger().info(f'Published action: {act.name}')
            except Exception:
                self.get_logger().warning('Non-JSON response from VLM')

    def _worker(self):
        while True:
            try:
                jpg = self.q_req.get(timeout=1.0)
            except Empty:
                continue

            # dry-run モード
            if _client is None:
                dummy = self.actions[0]
                self.q_res.put(json.dumps({
                    "anomaly":"pose_error",
                    "action":dummy['action'],
                    "target_id":dummy['target_id']
                }))
                continue

            b64 = base64.b64encode(jpg).decode('utf-8')
            messages = [
                {"role":"user","content":[
                    {"type":"text","text":self.prompt},
                    {"type":"text","text":
                        "Available actions:\n" +
                        "\n".join(
                            f"- {a['action']} (target_id={a['target_id']})"
                            for a in self.actions
                        )
                    },
                    {"type":"image_url",
                     "image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]

            try:
                res = _client.chat.completions.create(
                    model='gpt-4o-mini', max_tokens=60, messages=messages)
                self.q_res.put(res.choices[0].message.content.strip())
            except Exception as e:
                err = {"anomaly":"api_error","action":"","target_id":"","err":str(e)}
                self.q_res.put(json.dumps(err))


def main(args=None):
    parsed = parse_args()
    rclpy.init(args=args)
    node = VLMWatcher(parsed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
