import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy
from PIL import Image
import sys
import argparse
import os.path
from getpass import getpass
import time
import streamlit as st

#intro
logo = """\033[1;38m\033[1m
Stegano-graphy\033[1;m\033[0m 
                                                 \033[1m\033[37m\033[91m\033[37m\033[1;m\033[0m                      
"""

print(logo)

def slowprint(s):
    for c in s + '\n' :
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(10. / 100)
slowprint("\033[1m\033[1;33m [!] Loading...\n\n\033[1;m\033[0m ")
time.sleep(0)
# os.system('clear')

print(logo)

#choice
input("\033[1m Enter \033[91mS\033[1;m \033[1mto continue:\033[0m ")
# os.system("clear")
print(logo)

print("\n \033[1mOptions:\033[0m \n")
print(" [\033[1;38m01\033[1;m] Steganography          [\033[1;38m03\033[1;m] About")
print(" [\033[1;38m02\033[1;m] Cryptography           [\033[1;38m99\033[1;m] Exit")

steorcry = input(" \n\033[1m\033[1;33m [#]:> \033[1;m\033[0m")

#main
if steorcry == "01" or steorcry == "1":
	# os.system("clear")
	print(logo)

	print("\n \033[1mOptions:\033[0m \n")
	print(" [\033[1;38m01\033[1;m] Encode                 [\033[1;38m99\033[1;m] Exit")
	print(" [\033[1;38m02\033[1;m] Decode")


	choice1 = input(" \n\033[1m\033[1;33m [#]:> \033[1;m\033[0m")

	if choice1 == "1" or choice1 == "01":
		print("\n What do you want to Encode?")
		print(" Options:")
		print(" [01] Text")
		print(" [02] File")
		choice2 = input("\033[1m\033[1;33m [#]:> \033[1;m\033[0m")
		if choice2 == "1" or choice2 == "01":
			print("\n Do you want to keep a password? [Y/N] ")
			pas1 = input("\033[1m\033[1;33m [#]:> \033[1;m\033[0m")
			if pas1 == "y" or pas1 == "Y":
				print(" Enter text to encode.")
				msg = input (" \033[1m\033[1;33m [#]:> \033[1;m\033[0m")
				pthost = input(" Path to host file: ")
				print("")
				os.system("python stegano.py " + msg + " " + pthost + " -p")
			
			elif pas1 == "N" or pas1 == "n":
				print(" Enter text to encode.")
				msg1 = input (" \033[1m\033[1;33m [#]:> \033[1;m\033[0m")
				pthost1 = input(" Path to host file: ")
				print("")
				os.system("python stegano.py " + msg1 + " " + pthost1)	
			else:
				print("Invalid input found. Please re-run the script.")
			
		elif choice2 == "2" or choice2 == "02":
			print(" Do you want to keep a password? [Y/N] ")
			pas2 = input("\033[1m\033[1;33m [#]:> \033[1;m\033[0m")

			if pas2 == "y" or pas2 == "Y":
				victim = input(" Path to file to encode: ")
				host = input(" Path to host file: ")
				print("")
				os.system("python stegano.py " + victim + " " + host + " -p")

			elif pas2 == "N" or pas2 == "n":
				victim1 = input(" Path to file to encode: ")
				host1 = input(" Path to host file: ")
				print("")
				os.system("python stegano.py " + victim1 + " " + host1)

			else:
				print("Invalid input found. Please re-run the script.")

		else:
			print("Invalid input found. Please re-run the script.")


	elif choice1 == "2" or choice1 == "02":
		print("\n Does it have a password? [Y/N]")
		askp = input("\033[1m\033[1;33m [#]:> \033[1;m\033[0m")

		if askp == "Y" or askp == "y":
			decode_path = input("\n Path to file to decode: ")
			print("")
			os.system("python stegano.py " + decode_path + " -p")

		elif askp == "n" or askp == "N":
			decode_path1 = input("\n Path to file to decode: ")
			print("")
			os.system("python stegano.py " + decode_path1)
		else:
			print("Invalid input found. Please re-run the script.")

	
	elif choice1 == "99":
		os.system("exit()")

	else:
		print("Invalid input found. Please re-run the script.")

elif steorcry == "2" or steorcry == "02":
	# os.system("clear")
	os.system("python crypto.py")

elif steorcry == "3" or choice1 == "03":
		# os.system("clear")
		print(logo)
		print("		--> ABOUT <--")
		print("""\n
		Note:
		This 5HR3NOGRAPH tool was created for educational and personal purposes only. 
		The creators will not be held responsible for any violations of law caused by any
		means by anyone. Please use this tool at your own risk.

		What is steganography?
		Steganography is the practice of concealing a file, message, image, or video within
		another file, message, image, or video. In simple words- hiding media inside media.

		What is cryptography?
		Cryptography is the process of converting ordinaryplain text into unintelligible 
		text. In simple words- making a text uncomprehensible.

		5HR3NOGRAPH (Shre-no-graf) is a tool allowing you to perform both Steganography 
		and Cryptography. You can hide, encrypt text, hide files, media etc. It is highly 
		advised to encrypt text before hidning it for better security. 

			There are 3 main files in this directory:

			[01] stegano.py: Steganography code.
			[03] crypto.py: Cryptography code.
			[03] run.py: For begginers who are confused.

			[Please shuffle the characters in the variable 'main' on line 35 in file crypto.py
			for	unique encryption through 5HR3NOGRAPH Cryptography.]

			>>You can directly use crypto.py and stegano.py by using arguements.
			  Please refer to the documentation to know how.



		CONTACT ME: https://linktr.ee/5HR3D 
		
.
		Goodluck using 5HR3NOGRAPH.
		Thank you.\n""")

elif steorcry == "99":
	exit()

else:
	print("Invalid input found. Please re-run the script.")










