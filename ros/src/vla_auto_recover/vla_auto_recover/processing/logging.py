import time
import datetime
import json

def write_detail_log(self, event_type: str, detail_log: dict):
    """Write detailed log entry"""
    timestamp = time.time()
    log_entry = {
        'timestamp': timestamp,
        'datetime': datetime.fromtimestamp(timestamp).isoformat(),
        'event_type': event_type,
        'details': detail_log
    }    
    try:
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, indent=None) + '\n')
    except Exception as e:
        self.get_logger().error(f'Failed to write log: {e}')

    return None
