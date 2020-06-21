# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
account_sid = 'ACa45d6c6ebe3168ec793d21cb830fff30'
auth_token = 'fc7cac2b17f3dad32cc682008bd0f2ec'
client = Client(account_sid, auth_token)

call = client.calls.create(
    url='http://demo.twilio.com/docs/voice.xml',
    to='+917494943642',
    from_='+12517322595'
)

# print(call.sid)
