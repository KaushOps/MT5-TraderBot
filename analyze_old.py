import urllib.request
import json

req = urllib.request.Request('https://tmpfiles.org/dl/35091291/diary.jsonl', headers={'User-Agent': 'Mozilla/5.0'})
content = urllib.request.urlopen(req).read().decode('utf-8').strip().split('\n')

wins = 0
losses = 0
buys = 0
sells = 0

print(f"Total lines: {len(content)}")

for line in content:
    try:
        data = json.loads(line)
        act = str(data.get("action", ""))
        reason = str(data.get("reason", ""))
        rationale = str(data.get("rationale", ""))
        
        if act == "buy": buys += 1
        elif act == "sell": sells += 1
        elif act == "Position_Closed" or "SL_Hit" in act or "TP_Hit" in act or act == "risk_force_close":
            if "profit" in data:
                opt_profit = float(data["profit"])
                if opt_profit > 0: wins += 1
                else: losses += 1
    except:
        pass

total = wins + losses
if total > 0:
    print(f"Total trades resolved: {total} | Wins: {wins} | Losses: {losses} | WR: {wins/total*100:.1f}%")
else:
    print("No resolved trades found.")
