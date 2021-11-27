from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii
import base64

def base64_e(input):
	try:
		decoded_string = input
		encoded_string = base64.b64encode(decoded_string.encode('ascii'))
		return (encoded_string.decode('ascii'))
		# print()
	except:
		print(' Invalid Input')


def base64_d(input):
	try:
		encoded_string = input
		decoded_string = base64.b64decode(encoded_string.encode('ascii'))
		return (decoded_string.decode('ascii'))
		# print()
	except:
		print(' Invalid Input')


def encrypt(y, keyPair):
  pubKey = keyPair.publickey()
  pubKeyPEM = pubKey.exportKey()
  privKeyPEM = keyPair.exportKey()
  msg = y.encode('UTF-8')
  encryptor = PKCS1_OAEP.new(pubKey)
  encrypted = encryptor.encrypt(msg)
  return encrypted

def decrypt(keyPair, encrypted):
  decryptor = PKCS1_OAEP.new(keyPair)
  decrypted__ = decryptor.decrypt(encrypted)
  decrypted_ = decrypted__.decode('UTF-8')
  decrypted = base64_d(decrypted_)
  return decrypted

x = int (input())
keyPair = RSA.generate(x)
y_ = input()
y = base64_e(y_)

encrypted  = encrypt(y, keyPair)
encrypted = encrypted.decode('UTF-8')
print(encrypted)

decrypted = decrypt(keyPair, encrypted)
print(decrypted)
# print('Decrypted:', decrypted)