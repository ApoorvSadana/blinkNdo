# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
account_sid = 'ACace197a9331443444a52e549d343f96b'
auth_token = 'fa111419a6f4998c2b8ff33b23b6c083'
client = Client(account_sid, auth_token)

call = client.calls.create(
                        url='http://demo.twilio.com/docs/voice.xml',
                        to='+919818394095',
                        from_='+19282603727'
                    )

print(call.sid)
