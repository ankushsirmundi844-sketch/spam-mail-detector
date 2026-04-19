"""
Generates a realistic SMS spam dataset for demo purposes.
If you have the real UCI SMSSpamCollection file, place it as data/sms.tsv instead.
"""
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

SPAM = [
    "WINNER!! You have been selected to win a cash prize of £{n} CALL NOW {p}",
    "Congratulations! Your mobile has won £{n} prize. Call {p} to claim. Send STOP to opt out.",
    "URGENT! Your Mobile number has been awarded a £{n} Bonus Caller Prize. Call {p} now.",
    "FREE entry to win FA Cup final tkts. Text FA to 87121. Cost £{n}p/msg.",
    "You have won a Nokia 6610. Call {p} now to claim. Send STOP to 87239 to end.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 send to 87575. Cost: £{n}p/msg.",
    "Claim your FREE gift now! Text YES to {p} to receive your £{n} voucher. Ts&Cs apply.",
    "Win a trip to Bahamas! You are selected. Call {p} within 48 hours. Ref: HOLS{n}",
    "Hot singles in your area! Text MEET to {p}. Msgs cost £{n} ea. Stop: STOP.",
    "IMPORTANT: Your account will be suspended. Verify now or lose access permanently at http://bit.ly/{n}",
    "Naughty text messages! Txt CHAT to {p} £{n}/msg 2 stop txt STOP.",
    "You have an important message waiting. Call {p} now. Calls cost £{n}p/min.",
    "Your FREE loan of £{n} is approved. Call {p} to collect. Ts&Cs apply.",
    "CASH PRIZE: Draw shows you have won a top prize of £{n}. To collect call {p}.",
    "PRIVATE! Your 2003 Account Statement shows £{n} UNCLAIMED! Call {p} to claim.",
]

HAM = [
    "Hey, are you free this evening? We could grab dinner.",
    "Can you pick up some milk on your way home?",
    "Meeting has been moved to 3pm. See you there.",
    "Happy birthday! Hope you have a wonderful day!",
    "Just finished the report. Will send it over shortly.",
    "Are you coming to the party tonight? It starts at 8.",
    "Can we reschedule our call to tomorrow morning?",
    "Thanks for helping me yesterday, really appreciate it.",
    "Did you see the game last night? What a match!",
    "Running 10 mins late, sorry! Almost there.",
    "The package has been delivered to your door.",
    "Remember we have that dentist appointment tomorrow at 2pm.",
    "Just checking in, how are you feeling today?",
    "Lunch at the usual place? 12:30 works for me.",
    "Mom says dinner is ready at 7, don't be late.",
    "Can you send me the address again? I lost it.",
    "Just got out of the meeting, calling you in 5.",
    "Did you get my email? Let me know when you can talk.",
    "No worries, take your time. I'll be here.",
    "The kids are asking about the weekend trip. You in?",
]

msgs, labels = [], []
for _ in range(1500):
    t = random.choice(SPAM)
    filled = t.format(n=random.randint(100, 9999), p=str(random.randint(7000000000, 9999999999)))
    msgs.append(filled)
    labels.append("spam")

for _ in range(3974):
    base = random.choice(HAM)
    suffix = " " + "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=random.randint(0, 20)))
    msgs.append(base + suffix.rstrip())
    labels.append("ham")

df = pd.DataFrame({"label": labels, "message": msgs})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/sms.csv", index=False)
print(f"Dataset saved: {len(df)} messages ({labels.count('spam')} spam, {labels.count('ham')} ham)")
