import base64
import hashlib
from io import BytesIO
import os
import string
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import docx2txt
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy
from PIL import Image
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
import os.path
import streamlit as st
import pyffx
import string

# File Processing Pkgs
import pandas as pd
from PIL import Image
from PyPDF2 import PdfFileReader
import pdfplumber
from pydub import AudioSegment

# -----------------------------------------------STEGANOGRAPHY----------------------------------------------------


def derive_key(password, salt=None):
    if not salt:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )

    return [base64.urlsafe_b64encode(kdf.derive(password)), salt]


def encrypt_info(password, info):
    """Receives a password and a byte array. Returns a Fernet token."""
    password = bytes((password).encode("utf-8"))
    key, salt = derive_key(password)
    f = Fernet(key)
    token = f.encrypt(info)
    return bytes(salt) + bytes(token)


def decrypt_info(password, token, salt):
    """Receives a password and a Fernet token. Returns a byte array."""
    password = bytes((password).encode("utf-8"))
    key = derive_key(password, salt)[0]
    f = Fernet(key)
    info = f.decrypt(token)
    return info


MAGIC_NUMBER = b"stegv3"


class HostElement:
    """This class holds information about a host element."""

    def __init__(self, filename):
        self.filename = filename
        self.format = filename[-3:]
        self.header, self.data = get_file(filename)

    def save(self):
        self.filename = self.filename
        if self.format.lower() == "wav":
            sound = numpy.concatenate((self.header, self.data))
            sound.tofile(self.filename)
        elif self.format.lower() == "gif":
            gif = []
            for frame, palette in zip(self.data, self.header[0]):
                image = Image.fromarray(frame)
                image.putpalette(palette)
                gif.append(image)
            gif[0].save(
                self.filename,
                save_all=True,
                append_images=gif[1:],
                loop=0,
                duration=self.header[1],
            )
        else:
            if not self.filename.lower().endswith(("png", "bmp", "webp")):
                print("Host has a lossy format and will be converted to PNG.")
                st.info("Host has a lossy format and will be converted to PNG.")
                self.filename = self.filename[:-3] + "png"
            image = Image.fromarray(self.data)
            image.save(self.filename, lossless=True, minimize_size=True, optimize=True)
            st.markdown(
                HostElement.get_image_download_link(
                    image, self.filename, "Download " + self.filename
                ),
                unsafe_allow_html=True,
            )
            return image
        print("Information encoded in {}.".format(self.filename))
        st.success("Information encoded in {}.".format(self.filename))

    def get_image_download_link(img, filename, text):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = (
            f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
        )
        return href

    def insert_message(self, message, bits=2, parasite_filename=None, password=None):
        raw_message_len = len(message).to_bytes(4, "big")
        formatted_message = format_message(message, raw_message_len, parasite_filename)
        if password:
            formatted_message = encrypt_info(password, formatted_message)
        self.data = encode_message(self.data, formatted_message, bits)

    def read_message(self, password=None):
        print(self.data)
        msg = decode_message(self.data)
        print(msg)

        if password:
            try:
                print(password)
                salt = bytes(msg[:16])
                msg = decrypt_info(password, bytes(msg[16:]), salt)
            except:
                print("Wrong password.")
                st.warning("Wrong Password")
                return

        check_magic_number(msg)
        msg_len = int.from_bytes(bytes(msg[6:10]), "big")
        filename_len = int.from_bytes(bytes(msg[10:11]), "big")

        start = filename_len + 11
        end = start + msg_len
        end_filename = filename_len + 11
        if filename_len > 0:
            filename = bytes(msg[11:end_filename]).decode("utf-8")

        else:
            text = bytes(msg[start:end]).decode("utf-8")
            st.text("Decrpted Message in the file is")
            st.success(text)
            return

        with open(filename, "wb") as f:
            f.write(bytes(msg[start:end]))

        print("File {} succesfully extracted from {}".format(filename, self.filename))
        st.success(
            "File {} succesfully extracted from {}".format(filename, self.filename)
        )

    def free_space(self, bits=2):
        shape = self.data.shape
        self.data.shape = -1
        free = self.data.size * bits // 8
        self.data.shape = shape
        self.free = free
        return free

    def print_free_space(self, bits=2):
        free = self.free_space(bits)
        print(
            "File: {}, free: (bytes) {:,}, encoding: 4 bit".format(
                self.filename, free, bits
            )
        )
        st.success(
            "File: {}, free: (bytes) {:,}, encoding: 4 bit".format(
                self.filename, free, bits
            )
        )


def get_file(filename):
    """Returns data from file in a list with the header and raw data."""
    if filename.lower().endswith("wav"):
        content = numpy.fromfile(filename, dtype=numpy.uint8)
        content = content[:10000], content[10000:]
    elif filename.lower().endswith("gif"):
        image = Image.open(filename)
        frames = []
        palettes = []
        try:
            while True:
                frames.append(numpy.array(image))
                palettes.append(image.getpalette())
                image.seek(image.tell() + 1)
        except EOFError:
            pass
        content = [palettes, image.info["duration"]], numpy.asarray(frames)
    else:
        image = Image.open(filename)
        if image.mode != "RGB":
            image = image.convert("RGB")
        content = None, numpy.array(image)
    return content


def format_message(message, msg_len, filename=None):
    if not filename:  # text
        message = MAGIC_NUMBER + msg_len + (0).to_bytes(1, "big") + message
    else:
        filename = filename.encode("utf-8")
        filename_len = len(filename).to_bytes(1, "big")
        message = MAGIC_NUMBER + msg_len + filename_len + filename + message
    return message


def encode_message(host_data, message, bits):
    """Encodes the byte array in the image numpy array."""
    shape = host_data.shape
    host_data.shape = (-1,)  # convert to 1D
    uneven = 0
    divisor = 8 // bits

    st.info("Host dimension: {:,} bytes".format(host_data.size))
    st.info("Message size: {:,} bytes".format(len(message)))
    st.info("Maximum size: {:,} bytes".format(host_data.size // divisor))

    check_message_space(host_data.size // divisor, len(message))

    if (
        host_data.size % divisor != 0
    ):  # Hacky way to deal with pixel arrays that cannot be divided evenly
        uneven = 1
        original_size = host_data.size
        host_data = numpy.resize(
            host_data, host_data.size + (divisor - host_data.size % divisor)
        )

    msg = numpy.zeros(len(host_data) // divisor, dtype=numpy.uint8)

    msg[: len(message)] = list(message)

    host_data[: divisor * len(message)] &= 256 - 2 ** bits  # clear last bit(s)
    for i in range(divisor):
        host_data[i::divisor] |= msg >> bits * i & (
            2 ** bits - 1
        )  # copy bits to host_data

    operand = 0 if (bits == 1) else (16 if (bits == 2) else 32)
    host_data[0] = (host_data[0] & 207) | operand  # 5th and 6th bits = log_2(bits)

    if uneven:
        host_data = numpy.resize(host_data, original_size)

    host_data.shape = shape  # restore the 3D shape

    return host_data


def check_message_space(max_message_len, message_len):
    """Checks if there's enough space to write the message."""
    if max_message_len < message_len:
        st.warning("You have too few colors to store that message. Aborting.")
        exit(-1)
    else:
        print("Ok.")


def decode_message(host_data):
    """Decodes the image numpy array into a byte array."""
    host_data.shape = (-1,)  # convert to 1D
    bits = 2 ** ((host_data[0] & 48) >> 4)  # bits = 2 ^ (5th and 6th bits)
    divisor = 8 // bits

    if host_data.size % divisor != 0:
        host_data = numpy.resize(
            host_data, host_data.size + (divisor - host_data.size % divisor)
        )

    msg = numpy.zeros(len(host_data) // divisor, dtype=numpy.uint8)

    for i in range(divisor):
        msg |= (host_data[i::divisor] & (2 ** bits - 1)) << bits * i

    return msg


def check_magic_number(msg):
    if bytes(msg[0:6]) != MAGIC_NUMBER:
        print(bytes(msg[:6]))
        st.warning("ERROR! No encoded info found!")
        exit(-1)


# -----------------------------------------------CRYPTOGRAPHY-----------------------------------------------------

# 5HR3code
class crptoo:
    def shrecode_steg_e(self, message):
        main = """jO!l@JT<Eu+Am>z%*C-0sy,Wc"wt$Sa.Fh~D|x\X9?PG/nK`dU=q#LQ;8V:pr7'eBg 6[Y&b5oH^]i4N(f3}_{kZI2v1R)M"""
        newMessage = ""
        ekey = -19
        # message = st.text_input("Enter the Encoding text")
        for character in message:
            if character in main:
                position = main.find(character)
                newPosition = (position + ekey) % 95
                newCharacter = main[newPosition]
                newMessage += newCharacter
            else:
                newMessage += character
        print(newMessage)
        return newMessage

    def shrecode_steg_d(self, message):
        main = """jO!l@JT<Eu+Am>z%*C-0sy,Wc"wt$Sa.Fh~D|x\X9?PG/nK`dU=q#LQ;8V:pr7'eBg 6[Y&b5oH^]i4N(f3}_{kZI2v1R)M"""
        newMessage = ""
        dkey = +19
        # message = input(' \n\033[1m\n Enter Text: \033[0m')
        for character in message:
            if character in main:
                position = main.find(character)
                newPosition = (position + dkey) % 95
                newCharacter = main[newPosition]
                newMessage += newCharacter
            else:
                newMessage += character

        return newMessage

    def shrecode_e(self, message):
        main = """jO!l@JT<Eu+Am>z%*C-0sy,Wc"wt$Sa.Fh~D|x\X9?PG/nK`dU=q#LQ;8V:pr7'eBg 6[Y&b5oH^]i4N(f3}_{kZI2v1R)M"""
        newMessage = "\n  "
        ekey = -19
        # message = st.text_input("Enter the Encoding text")
        for character in message:
            if character in main:
                position = main.find(character)
                newPosition = (position + ekey) % 95
                newCharacter = main[newPosition]
                newMessage += newCharacter
            else:
                newMessage += character
        st.text("Encrypted Text")
        st.success(newMessage)

    def shrecode_d(self, message):
        main = """jO!l@JT<Eu+Am>z%*C-0sy,Wc"wt$Sa.Fh~D|x\X9?PG/nK`dU=q#LQ;8V:pr7'eBg 6[Y&b5oH^]i4N(f3}_{kZI2v1R)M"""
        newMessage = "\n  "
        dkey = +19
        # message = input(' \n\033[1m\n Enter Text: \033[0m')
        for character in message:
            if character in main:
                position = main.find(character)
                newPosition = (position + dkey) % 95
                newCharacter = main[newPosition]
                newMessage += newCharacter
            else:
                newMessage += character
        st.text("Decrypted Text")
        st.success(newMessage)

    # hashes encryption
    def md5_e(Self, mystring):
        # mystring = input(' Text to encrypt: ')
        hash_md5 = hashlib.md5(mystring.encode())
        st.text("Encrypted Text")
        st.success(hash_md5.hexdigest())

    def sha1_e(self, mystring):
        # mystring = input(' Text to encrypt: ')
        hash_sha1 = hashlib.sha1(mystring.encode())
        st.text("Encrypted Text")
        st.success(hash_sha1.hexdigest())

    def sha256_e(self, mystring):
        # mystring = input(' Text to encrypt: ')
        hash_sha256 = hashlib.sha256(mystring.encode())
        st.text("Encrypted Text")
        st.success(hash_sha256.hexdigest())

    def sha224_e(self, mystring):
        # mystring = input(' Text to encrypt: ')
        hash_sha224 = hashlib.sha224(mystring.encode())
        st.text("Encrypted Text")
        st.success(hash_sha224.hexdigest())

    def sha512_e(self, mystring):
        # mystring = input(' Text to encrypt: ')
        hash_sha512 = hashlib.sha512(mystring.encode())
        st.text("Encrypted Text")
        st.success(hash_sha512.hexdigest())

    # ----------------------base64----------------------------------------
    def base64_e(self, decoded_string):
        try:
            # decoded_string = input('String To Encode: ')
            encoded_string = base64.b64encode(decoded_string.encode("ascii"))
            st.success(encoded_string.decode("ascii"))
        except:
            st.warning("Invalid Input")

    def base64_d(self, encoded_string):
        try:
            # encoded_string = input('Encoded String : ')
            decoded_string = base64.b64decode(encoded_string.encode("ascii"))
            st.success(decoded_string.decode("ascii"))
        except:
            st.warning("Invalid Input")

    # -------------------Format-Preserving-Encryption----------------------------

    def fpenum_e(self, nume):
        try:
            # nume = input(" Enter text to encrypt: ")
            numelen = len(nume)
            e = pyffx.Integer(b"(5hr3d)", length=int(numelen))
            f = e.encrypt(nume)
            st.success(f)
        except:
            st.warning(" Invalid Input.")

    def fpenum_d(self, numd):
        try:
            # numd = input(" Enter text to decrypt: ")
            numdlen = len(numd)
            e = pyffx.Integer(b"(5hr3d)", length=int(numdlen))
            f = e.decrypt(numd)
            st.success(f)
        except:
            st.warning(" Invalid Input.")

    def fpealp_e(self, alpe):
        try:
            # alpe = input(" Enter text to encrypt: ")
            alpelen = len(alpe)
            e = pyffx.String(
                b"(5hr3d)", alphabet="abcdefghijklmnopqrstuvwxyz", length=int(alpelen)
            )
            f = e.encrypt(alpe)
            st.success(f)
        except:
            st.warning(" Invalid Input.")

    def fpealp_d(self, alpd):
        try:
            # alpd = input(" Enter text to decrypt: ")
            alpdlen = len(alpd)
            e = pyffx.String(
                b"(5hr3d)", alphabet="abcdefghijklmnopqrstuvwxyz", length=int(alpdlen)
            )
            f = e.decrypt(alpd)
            st.success(f)
        except:
            st.warning(" Invalid Input.")

    # --------------------------------------------------------------------
    def hill_e(self, message_, n, s):
        main = string.ascii_lowercase

        def generate_key(n, s):
            s = s.replace(" ", "")
            s = s.lower()

            key_matrix = ["" for i in range(n)]
            i = 0
            j = 0
            for c in s:
                if c in main:
                    key_matrix[i] += c
                    j += 1
                    if j > n - 1:
                        i += 1
                        j = 0
            st.info("The key matrix " + "(" + str(n) + "x" + str(n) + ") is:")
            st.write(key_matrix)

            key_num_matrix = []
            for i in key_matrix:
                sub_array = []
                for j in range(n):
                    sub_array.append(ord(i[j]) - ord("a"))
                key_num_matrix.append(sub_array)

            for i in key_num_matrix:
                st.write(i)
            return key_num_matrix

        def message_matrix(s, n):
            s = s.replace(" ", "")
            s = s.lower()
            final_matrix = []
            if len(s) % n != 0:
                # z is the bogus word
                while len(s) % n != 0:
                    s = s + "z"

            st.info("Converted plain_text for encryption: ")
            st.write(s)
            for k in range(len(s) // n):
                message_matrix = []
                for i in range(n):
                    sub = []
                    for j in range(1):
                        sub.append(ord(s[i + (n * k)]) - ord("a"))
                    message_matrix.append(sub)
                final_matrix.append(message_matrix)
            st.info("The column matrices of plain text in numbers are:  ")
            for i in final_matrix:
                st.write(i)
            return final_matrix

        # Function to get cofactor of
        # mat[p][q] in temp[][]
        # passing the key matrix as 'mat' to check for invertibility
        def getCofactor(mat, temp, p, q, n):
            i = 0
            j = 0

            # Looping for each element
            # of the matrix
            for row in range(n):
                for col in range(n):
                    # Copying into temporary matrix only those element
                    # which are not in given row and column
                    if row != p and col != q:
                        temp[i][j] = mat[row][col]
                        j += 1

                        # Row is filled, so increase
                        # row index and reset col index
                        if j == n - 1:
                            j = 0
                            i += 1

        # Recursive function for finding determinant of matrix.
        # n is current dimension of mat[][].
        def determinantOfMatrix(mat, n):
            D = 0  # Initialize result
            # Base case : if matrix
            # contains single element
            if n == 1:
                return mat[0][0]

            # To store cofactors
            temp = [[0 for x in range(n)] for y in range(n)]

            sign = 1  # To store sign multiplier

            # Iterate for each
            # element of first row
            for f in range(n):

                # Getting Cofactor of mat[0][f]
                getCofactor(mat, temp, 0, f, n)
                D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1)

                # terms are to be added with alternate sign
                sign = -sign
            return D

        def isInvertible(mat, n):
            if determinantOfMatrix(mat, n) != 0:
                return True
            else:
                return False

        def multiply_and_convert(key, message):

            # multiplying matrices
            # resultant must have:
            # rows = numbers of rows in message matrix
            # columns = number of columns in key matrix
            res_num = [[0 for x in range(len(message[0]))] for y in range(len(key))]

            for i in range(len(key)):
                for j in range(len(message[0])):
                    for k in range(len(message)):
                        # resulted number matrix
                        res_num[i][j] += key[i][k] * message[k][j]

            res_alpha = [["" for x in range(len(message[0]))] for y in range(len(key))]
            # getting the alphabets from the numbers
            # according to the logic of hill ciipher
            for i in range(len(key)):
                for j in range(len(message[0])):
                    # resultant alphabet matrix
                    res_alpha[i][j] += chr((res_num[i][j] % 26) + 97)
            return res_alpha

        # implementing all logic and calling function
        key = generate_key(n, s)

        # check for invertability here
        if isInvertible(key, len(key)):
            st.info("Yes it is invertable and can be decrypted")
        else:
            st.info("No it is not invertable and cannot be decrypted")

        message = message_matrix(message_, n)
        final_message = ""
        for i in message:
            sub = multiply_and_convert(key, i)
            for j in sub:
                for k in j:
                    final_message += k
        st.info("plain message: ")
        st.write(message_)
        st.text("final encrypted message")
        st.success(final_message)

    # --------------------------------------------------------------------

    def hill_d(self, message_, n, s):

        main = string.ascii_lowercase

        def generate_key(n, s):
            s = s.replace(" ", "")
            s = s.lower()

            key_matrix = ["" for i in range(n)]
            i = 0
            j = 0
            for c in s:
                if c in main:
                    key_matrix[i] += c
                    j += 1
                    if j > n - 1:
                        i += 1
                        j = 0
            st.info("The key matrix " + "(" + str(n) + "x" + str(n) + ") is:")
            st.write(key_matrix)

            key_num_matrix = []
            for i in key_matrix:
                sub_array = []
                for j in range(n):
                    sub_array.append(ord(i[j]) - ord("a"))
                key_num_matrix.append(sub_array)

            for i in key_num_matrix:
                st.write(i)
            return key_num_matrix

        def modInverse(a, m):
            a = a % m
            for x in range(1, m):
                if (a * x) % m == 1:
                    return x
            return 1

        def method(a, m):
            if a > 0:
                return a % m
            else:
                k = (abs(a) // m) + 1
            return method(a + k * m, m)

        def message_matrix(s, n):
            s = s.replace(" ", "")
            s = s.lower()
            final_matrix = []
            if len(s) % n != 0:
                # may be negative also
                for i in range(abs(len(s) % n)):
                    # z is the bogus word
                    s = s + "z"
            st.info("Converted cipher_text for decryption: ")
            st.write(s)
            for k in range(len(s) // n):
                message_matrix = []
                for i in range(n):
                    sub = []
                    for j in range(1):
                        sub.append(ord(s[i + (n * k)]) - ord("a"))
                    message_matrix.append(sub)
                final_matrix.append(message_matrix)
            st.info("The column matrices of plain text in numbers are:  ")
            for i in final_matrix:
                st.write(i)
            return final_matrix

        def multiply_and_convert(key, message):

            # multiplying matrices
            # resultant must have:
            # rows = numbers of rows in message matrix
            # columns = number of columns in key matrix
            res_num = [[0 for x in range(len(message[0]))] for y in range(len(key))]

            for i in range(len(key)):
                for j in range(len(message[0])):
                    for k in range(len(message)):
                        # resulted number matrix
                        res_num[i][j] += key[i][k] * message[k][j]

            res_alpha = [["" for x in range(len(message[0]))] for y in range(len(key))]
            # getting the alphabets from the numbers
            # according to the logic of hill ciipher
            for i in range(len(key)):
                for j in range(len(message[0])):
                    # resultant alphabet matrix
                    res_alpha[i][j] += chr((res_num[i][j] % 26) + 97)

            return res_alpha

        key_matrix = generate_key(n, s)
        A = np.array(key_matrix)
        det = np.linalg.det(A)
        adjoint = det * np.linalg.inv(A)

        if det != 0:
            convert_det = modInverse(int(det), 26)
            adjoint = adjoint.tolist()
            st.info("Adjoint Matrix before modulo26 operation: ")
            for i in adjoint:
                st.write(i)
            st.write(convert_det)

            # applying modulo 26 to all elements in adjoint matrix
            for i in range(len(adjoint)):
                for j in range(len(adjoint[i])):
                    adjoint[i][j] = round(adjoint[i][j])
                    adjoint[i][j] = method(adjoint[i][j], 26)
            st.info("Adjoint Matrix after modulo26 operation: ")
            for i in adjoint:
                st.write(i)

            # modulo is applied to inverse of determinant and
            # multiplied to all elements in the adjoint matrix
            # to form inverse matrix
            adjoint = np.array(adjoint)
            inverse = convert_det * adjoint

            inverse = inverse.tolist()
            for i in range(len(inverse)):
                for j in range(len(inverse[i])):
                    inverse[i][j] = inverse[i][j] % 26

            st.info("Inverse matrix after applying modulo26 operation: ")
            for i in inverse:
                st.write(i)

            message = message_matrix(message_, n)
            plain_text = ""
            for i in message:
                sub = multiply_and_convert(inverse, i)
                for j in sub:
                    for k in j:
                        plain_text += k

            st.text("plain message")
            st.success(plain_text)
        else:
            st.warning("Matrix cannot be inverted")

    # --------------------------playfair----------------------------------
    def playfair_e(self, message, key):
        def key_generation(key):
            # initializing all and generating key_matrix
            main = string.ascii_lowercase.replace("j", ".")
            # convert all alphabets to lower
            key = key.lower()

            key_matrix = ["" for i in range(5)]
            # if we have spaces in key, those are ignored automatically
            i = 0
            j = 0
            for c in key:
                if c in main:
                    # putting into matrix
                    key_matrix[i] += c

                    # to make sure repeated characters in key
                    # doesnt include in the key_matrix, we replace the
                    # alphabet into . in the main, whenever comes in iteration
                    main = main.replace(c, ".")
                    # counting column change
                    j += 1
                    # if column count exceeds 5
                    if j > 4:
                        # row count is increased
                        i += 1
                        # column count is set again to zero
                        j = 0

            # to place other alphabets in the key_matrix
            # the i and j values returned from the previous loop
            # are again used in this loop, continuing the values in them
            for c in main:
                if c != ".":
                    key_matrix[i] += c

                    j += 1
                    if j > 4:
                        i += 1
                        j = 0

            return key_matrix

        # Now plaintext is to be converted into cipher text

        def conversion(plain_text):
            # seggrigating the maeesage into pairs
            plain_text_pairs = []
            # replacing repeated characters in pair with other letter, x
            cipher_text_pairs = []

            # remove spaces
            plain_text = plain_text.replace(" ", "")
            # convert to lower case
            plain_text = plain_text.lower()

            # RULE1: if both letters in the pair are same or one letter is left at last,
            # replace second letter with x or add x, else continue with normal pairing

            i = 0
            # let plain_text be abhi
            while i < len(plain_text):
                # i=0,1,2,3
                a = plain_text[i]
                b = ""

                if (i + 1) == len(plain_text):
                    # if the chosen letter is last and doesnt have pair
                    # then the pai will be x
                    b = "x"
                else:
                    # else the next letter will be pair with the previous letter
                    b = plain_text[i + 1]

                if a != b:
                    plain_text_pairs.append(a + b)
                    # if not equal then leave the next letter,
                    # as it became pair with previous alphabet
                    i += 2
                else:
                    plain_text_pairs.append(a + "x")
                    # else dont leave the next letter and put x
                    # in place of repeated letter and conitnue with the next letter
                    # which is repeated (according to algo)
                    i += 1

            st.info("plain text pairs: ")
            st.write(plain_text_pairs)

            for pair in plain_text_pairs:
                # RULE2: if the letters are in the same row, replace them with
                # letters to their immediate right respectively
                flag = False
                for row in key_matrix:
                    if pair[0] in row and pair[1] in row:
                        # find will return index of a letter in string
                        j0 = row.find(pair[0])
                        j1 = row.find(pair[1])
                        cipher_text_pair = row[(j0 + 1) % 5] + row[(j1 + 1) % 5]
                        cipher_text_pairs.append(cipher_text_pair)
                        flag = True
                if flag:
                    continue

                # RULE3: if the letters are in the same column, replace them with
                # letters to their immediate below respectively

                for j in range(5):
                    col = "".join([key_matrix[i][j] for i in range(5)])
                    if pair[0] in col and pair[1] in col:
                        # find will return index of a letter in string
                        i0 = col.find(pair[0])
                        i1 = col.find(pair[1])
                        cipher_text_pair = col[(i0 + 1) % 5] + col[(i1 + 1) % 5]
                        cipher_text_pairs.append(cipher_text_pair)
                        flag = True
                if flag:
                    continue
                # RULE:4 if letters are not on the same row or column,
                # replace with the letters on the same row respectively but
                # at the other pair of corners of rectangle,
                # which is defined by the original pair

                i0 = 0
                i1 = 0
                j0 = 0
                j1 = 0

                for i in range(5):
                    row = key_matrix[i]
                    if pair[0] in row:
                        i0 = i
                        j0 = row.find(pair[0])
                        if pair[1] in row:
                            i1 = i
                            j1 = row.find(pair[1])
                cipher_text_pair = key_matrix[i0][j1] + key_matrix[i1][j0]
                cipher_text_pairs.append(cipher_text_pair)

            st.info("cipher text pairs: ")
            st.write(cipher_text_pairs)
            # final statements
            st.info("plain text: ")
            st.write(plain_text)
            st.text("cipher text")
            st.success("".join(cipher_text_pairs))

        # calling first function
        key_matrix = key_generation(key)
        st.info("Key Matrix for encryption:")
        st.write(key_matrix)

        # calling second function
        conversion(message)

    # ---------------------------------playfair_d--------------------------------

    def playfair_d(self, message, key):
        def key_generation(key):
            # initializing all and generating key_matrix
            main = string.ascii_lowercase.replace("j", ".")
            # convert all alphabets to lower
            key = key.lower()

            key_matrix = ["" for i in range(5)]
            # if we have spaces in key, those are ignored automatically
            i = 0
            j = 0
            for c in key:
                if c in main:
                    # putting into matrix
                    key_matrix[i] += c

                    # to make sure repeated characters in key
                    # doesnt include in the key_matrix, we replace the
                    # alphabet into . in the main, whenever comes in iteration
                    main = main.replace(c, ".")
                    # counting column change
                    j += 1
                    # if column count exceeds 5
                    if j > 4:
                        # row count is increased
                        i += 1
                        # column count is set again to zero
                        j = 0

            # to place other alphabets in the key_matrix
            # the i and j values returned from the previous loop
            # are again used in this loop, continuing the values in them
            for c in main:
                if c != ".":
                    key_matrix[i] += c

                    j += 1
                    if j > 4:
                        i += 1
                        j = 0

            return key_matrix

        # Now ciphertext is to be converted into plaintext

        def conversion(cipher_text):
            # seggrigating the maeesage into pairs
            plain_text_pairs = []
            # replacing repeated characters in pair with other letter, x
            cipher_text_pairs = []

            # convert to lower case
            cipiher_text = cipher_text.lower()

            # RULE1: if both letters in the pair are same or one letter is left at last,
            # replace second letter with x or add x, else continue with normal pairing

            i = 0
            while i < len(cipher_text):
                # i=0,1,2,3
                a = cipher_text[i]
                b = cipher_text[i + 1]

                cipher_text_pairs.append(a + b)
                # else dont leave the next letter and put x
                # in place of repeated letter and conitnue with the next letter
                # which is repeated (according to algo)
                i += 2

            st.info("cipher text pairs")
            st.write(cipher_text_pairs)

            for pair in cipher_text_pairs:
                # RULE2: if the letters are in the same row, replace them with
                # letters to their immediate right respectively
                flag = False
                for row in key_matrix:
                    if pair[0] in row and pair[1] in row:
                        # find will return index of a letter in string
                        j0 = row.find(pair[0])
                        j1 = row.find(pair[1])
                        # same as reverse
                        # instead of -1 we are doing +4 as it is modulo 5
                        plain_text_pair = row[(j0 + 4) % 5] + row[(j1 + 4) % 5]
                        plain_text_pairs.append(plain_text_pair)
                        flag = True
                if flag:
                    continue

                # RULE3: if the letters are in the same column, replace them with
                # letters to their immediate below respectively

                for j in range(5):
                    col = "".join([key_matrix[i][j] for i in range(5)])
                    if pair[0] in col and pair[1] in col:
                        # find will return index of a letter in string
                        i0 = col.find(pair[0])
                        i1 = col.find(pair[1])
                        # same as reverse
                        # instead of -1 we are doing +4 as it is modulo 5
                        plain_text_pair = col[(i0 + 4) % 5] + col[(i1 + 4) % 5]
                        plain_text_pairs.append(plain_text_pair)
                        flag = True
                if flag:
                    continue
                # RULE:4 if letters are not on the same row or column,
                # replace with the letters on the same row respectively but
                # at the other pair of corners of rectangle,
                # which is defined by the original pair

                i0 = 0
                i1 = 0
                j0 = 0
                j1 = 0

                for i in range(5):
                    row = key_matrix[i]
                    if pair[0] in row:
                        i0 = i
                        j0 = row.find(pair[0])
                    if pair[1] in row:
                        i1 = i
                        j1 = row.find(pair[1])
                plain_text_pair = key_matrix[i0][j1] + key_matrix[i1][j0]
                plain_text_pairs.append(plain_text_pair)

            st.info("plain text pairs: ")
            st.write(plain_text_pairs)
            # final statements

            st.write("cipher text: ")
            st.info("".join(cipher_text_pairs))
            st.text("plain text (message)")
            st.success("".join(plain_text_pairs))

        # calling first function
        key_matrix = key_generation(key)
        st.info("Key Matrix for encryption:")
        st.write(key_matrix)

        # calling second function
        conversion(message)

    # ---------------------------railfence_e-------------------------------------

    def railfence_e(self, message, key):

        # this function is to get the desired sequence
        def sequence(n):
            arr = []
            i = 0
            # creating the sequence required for
            # implementing railfence cipher
            # the sequence is stored in array
            while i < n - 1:
                arr.append(i)
                i += 1
            while i > 0:
                arr.append(i)
                i -= 1
            return arr

        # this is to implement the logic
        def railfence(s, n):
            # converting into lower cases
            s = s.lower()

            # If you want to remove spaces,
            # you can uncomment this
            # s=s.replace(" ","")

            # returning the sequence here
            L = sequence(n)
            st.write("The raw sequence of indices: ", L)

            # storing L in temp for reducing additions in further steps
            temp = L

            # adjustments
            while len(s) > len(L):
                L = L + temp

            # removing the extra last indices
            for i in range(len(L) - len(s)):
                L.pop()
            st.write("The row indices of the characters in the given string: ", L)

            st.write("Transformed message for encryption: ", s)

            # converting into cipher text
            num = 0
            cipher_text = ""
            while num < n:
                for i in range(L.count(num)):
                    # adding characters according to
                    # indices to get cipher text
                    cipher_text = cipher_text + s[L.index(num)]
                    L[L.index(num)] = n
                num += 1
            st.text("The cipher text is")
            st.success(cipher_text)

        railfence(message, key)

    # ---------------------------railfence_d-----------------------------

    def railfence_d(self, message, key):
        # this function is to get the desired sequence
        def sequence(n):
            arr = []
            i = 0
            # creating the sequence required for
            # implementing railfence cipher
            # the sequence is stored in array
            while i < n - 1:
                arr.append(i)
                i += 1
            while i > 0:
                arr.append(i)
                i -= 1
            return arr

        # this is to implement the logic
        def railfence(cipher_text, n):
            # converting into lower cases
            cipher_text = cipher_text.lower()

            # If you want to remove spaces,
            # you can uncomment this
            # s=s.replace(" ","")

            # returning the sequence here
            L = sequence(n)
            st.write("The raw sequence of indices: ", L)

            # storing L in temp for reducing additions in further steps
            # if not stored and used as below, the while loop
            # will create L of excess length
            temp = L

            # adjustments
            while len(cipher_text) > len(L):
                L = L + temp

            # removing the extra last indices
            for i in range(len(L) - len(cipher_text)):
                L.pop()

            # storing L.sort() in temp1
            temp1 = sorted(L)

            st.write("The row indices of the characters in the cipher string: ", L)

            st.write("The row indices of the characters in the plain string: ", temp1)

            st.write("Transformed message for decryption: ", cipher_text)

            # converting into plain text
            plain_text = ""
            for i in L:
                # k is index of particular character in the cipher text
                # k's value changes in such a way that the order of change
                # in k's value is same as plaintext order
                k = temp1.index(i)
                temp1[k] = n
                plain_text += cipher_text[k]

            st.text("The cipher text is")
            st.success(plain_text)

        railfence(message, key)

    # -------------------vernam_e------------------------------------------

    def vernam_e(self, message, key):

        # function to apply algo of vernam cipher
        def vernam(plain_text, key):

            # convert into lower cases and remove spaces

            plain_text = plain_text.replace(" ", "")
            key = key.replace(" ", "")
            plain_text = plain_text.lower()
            key = key.lower()

            # conditional statements
            if len(plain_text) != len(key):
                st.info("Lengths are different")

            else:
                cipher_text = ""

                # iterating through the length
                for i in range(len(plain_text)):
                    k1 = ord(plain_text[i]) - 97
                    k2 = ord(key[i]) - 97
                    s = chr((k1 + k2) % 26 + 97)
                    cipher_text += s
                st.text("Enrypted message is: ")
                st.success(cipher_text)

        vernam(message, key)

    # ---------------------------vernam_d---------------------------------

    def vernam_d(self, message, key):

        # function to apply algo of vernam cipher
        def vernam(cipher_text, key):

            # convert into lower cases and remove spaces
            cipher_text = cipher_text.lower()
            key = key.lower()
            cipher_text = cipher_text.replace(" ", "")
            key = key.replace(" ", "")

            plain_text = ""

            # iterating through the length
            for i in range(len(cipher_text)):
                k1 = ord(cipher_text[i]) - 97
                k2 = ord(key[i]) - 97
                s = chr((((k1 - k2) + 26) % 26) + 97)
                plain_text += s
            st.text("Decrypted message is: ")
            st.success(plain_text)

        vernam(message, key)

    # ----------------------------vigenere_e--------------------------------
    def vigenere_e(self, message, key):

        main = string.ascii_lowercase

        def conversion(plain_text, key):
            index = 0
            cipher_text = ""

            # convert into lower case
            plain_text = plain_text.lower()
            key = key.lower()

            # For generating key, the given keyword is repeated
            # in a circular manner until it matches the length of
            # the plain text.
            for c in plain_text:
                if c in main:
                    # to get the number corresponding to the alphabet
                    off = ord(key[index]) - ord("a")

                    # implementing algo logic here
                    encrypt_num = (ord(c) - ord("a") + off) % 26
                    encrypt = chr(encrypt_num + ord("a"))

                    # adding into cipher text to get the encrypted message
                    cipher_text += encrypt

                    # for cyclic rotation in generating key from keyword
                    index = (index + 1) % len(key)
                # to not to change spaces or any other special
                # characters in their positions
                else:
                    cipher_text += c
            st.text("cipher text")
            st.success(cipher_text)

        # calling function
        conversion(message, key)

    # ----------------------------vigenere_d--------------------------------
    def vigenere_d(self, message, key):
        main = string.ascii_lowercase

        def conversion(cipher_text, key):
            index = 0
            plain_text = ""

            # convert into lower case
            cipher_text = cipher_text.lower()
            key = key.lower()

            for c in cipher_text:
                if c in main:
                    # to get the number corresponding to the alphabet
                    off = ord(key[index]) - ord("a")

                    positive_off = 26 - off
                    decrypt = chr((ord(c) - ord("a") + positive_off) % 26 + ord("a"))

                    # adding into plain text to get the decrypted messag
                    plain_text += decrypt

                    # for cyclic rotation in generating key from keyword
                    index = (index + 1) % len(key)
                else:
                    plain_text += c

            st.text("plain text")
            st.success(plain_text)

        # calling function
        conversion(message, key)

    # ---------------------------caesar-----------------------------------
    def caesar_e(self, message, key):
        def encrypt(text, s):

            # Cipher(n) = De-cipher(26-n)
            s = s
            text = text.replace(" ", "")
            result = ""  # empty string
            for i in range(len(text)):
                char = text[i]
                if char.isupper():  # if the text[i] is in upper case
                    result = result + chr((ord(char) + s - 65) % 26 + 65)
                else:
                    result = result + chr((ord(char) + s - 97) % 26 + 97)
            return result

        st.text("Encoded word in Caeser cipher is")
        st.success(encrypt(message, key))

    def caesar_d(self, message, key):
        def decrypt(text, s):

            # Cipher(n) = De-cipher(26-n)
            s = 26 - s

            result = ""  # empty string
            for i in range(len(text)):
                char = text[i]
                if char.isupper():  # if the text[i] is in upper case
                    result = result + chr((ord(char) + s - 65) % 26 + 65)
                else:
                    result = result + chr((ord(char) + s - 97) % 26 + 97)
            return result

        st.text("Encoded word in Caeser cipher is")
        st.success(decrypt(message, key))


# -----------------------------------------------FILE READER------------------------------------------------------

def base64_cry_e(input):
	try:
		decoded_string = input
		encoded_string = base64.b64encode(decoded_string.encode('ascii'))
		return (encoded_string.decode('ascii'))
		# print()
	except:
		print(' Invalid Input')


def base64_cry_d(input):
	try:
		encoded_string = input
		decoded_string = base64.b64decode(encoded_string.encode('ascii'))
		return (decoded_string.decode('ascii'))
		# print()
	except:
		print(' Invalid Input')


def encrypt__(y, keyPair):
  pubKey = keyPair.publickey()
  pubKeyPEM = pubKey.exportKey()
  privKeyPEM = keyPair.exportKey()
  msg = y.encode('UTF-8')
  encryptor = PKCS1_OAEP.new(pubKey)
  encrypted = encryptor.encrypt(msg)
  return encrypted

def decrypt__(keyPair, encrypted):
  decryptor = PKCS1_OAEP.new(keyPair)
  decrypted__ = decryptor.decrypt(encrypted)
  decrypted_ = decrypted__.decode('UTF-8')
  decrypted = base64_cry_d(decrypted_)
  return decrypted

def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


def read_pdf_with_pdfplumber(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()


# import fitz  # this is pymupdf

# def read_pdf_with_fitz(file):
# 	with fitz.open(file) as doc:
# 		text = ""
# 		for page in doc:
# 			text += page.getText()
# 		return text


def load_image(image_file):
    img = Image.open(image_file)
    return img


def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return pages


class FileDownloader(object):
    def __init__(self, data, filename="myfile", file_ext="txt"):
        super(FileDownloader, self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}_{}_.{}".format(self.filename, timestr, self.file_ext)
        st.markdown("#### Download File ###")
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here to download the password key for Stegofile!!</a>'
        st.markdown(href, unsafe_allow_html=True)


def main():
    # https://www.geeksforgeeks.org/encrypt-and-decrypt-image-using-python/
    st.title("Hybrid approach to secure data combining Cryptography and Steganography")
    cp = crptoo()
    menu = ["Home", "Stagenography", "Crpytography"]
    submenu = ["Text", "File"]
    ch = ["Encode", "Decode"]
    crypt = ["HYBRID", "SELF", "Base64", "FPE", "Hashes", "Ciphers", "Info for nerds"]
    hashes = ["MD5", "SHA1", "SHA256", "SHA224", "SHA512"]
    fp = ["Numeric", "Alphabetic"]
    cipher = [
        "Hill Cipher",
        "Playfair cipher",
        "Rail-fence cipher",
        "Vernam cipher",
        "Vigenere cipher",
        "Caesar cipher",
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

        uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
        base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

    elif choice == "Stagenography":
        cho = st.selectbox("Operation Type", ch)

        if cho == "Encode":
            x = st.file_uploader(
                "Upload the Host file", type=["png", "gif", "wav", "jpg"]
            )

            if x is not None:
                file_details = {
                    "Filename": x.name,
                    "FileType": x.type,
                    "FileSize": x.size,
                }
                st.write(file_details)
                with open(os.path.join("tempDir", "host_file_" + x.name), "wb") as f:
                    f.write(x.getbuffer())

                host_path = "tempDir/host_file_" + x.name
                host = HostElement(host_path)
                # st.success("Saved File")
                if x.name[-3:] == "png":
                    if st.button("Show image"):
                        st.image(x)

                elif x.name[-3:] == "wav":
                    if st.button("Play Audio"):
                        audio_bytes = x.read()
                        st.audio(audio_bytes, format="audio/wav")

                elif x.name[-3:] == "gif":
                    if st.button("Pla video"):
                        contents = x.read()
                        data_url = base64.b64encode(contents).decode("utf-8")
                        x.close()

                        st.markdown(
                            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                            unsafe_allow_html=True,
                        )

                st.text("What do you want to Encode?")
                choice1 = st.selectbox("Options", submenu)
                if choice1 == "File":
                    file = st.file_uploader(
                        "Upload Parasite File",
                        type=["png", "jpeg", "jpg", "wav", "txt", "pdf", "docx", "gif"],
                    )
                    if file is not None:

                        # To See Details
                        # st.write(type(image_file))
                        # st.write(dir(image_file))
                        file_details = {
                            "Filename": file.name,
                            "FileType": file.type,
                            "FileSize": file.size,
                        }
                        st.write(file_details)
                        with open(
                            os.path.join("tempDir", "parasite_file_" + file.name), "wb"
                        ) as f:
                            f.write(file.getbuffer())

                        with open("tempDir/parasite_file_" + file.name, "rb") as myfile:
                            message = myfile.read()
                            # print(message)

                        if file.name[-3:] == "png" or file.name[-3:] == "jpg":
                            if st.button("Show Parasite image"):
                                st.image(file)

                        elif file.name[-3:] == "wav":
                            if st.button("Play Audio"):
                                audio_bytes = file.read()
                                st.audio(audio_bytes, format="audio/wav")

                        elif file.name[-3:] == "gif":
                            if st.button("Pla video"):
                                contents = file.read()
                                data_url = base64.b64encode(contents).decode("utf-8")
                                file.close()

                                st.markdown(
                                    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                                    unsafe_allow_html=True,
                                )

                        elif file.name[-3:] == "pdf":
                            try:
                                with pdfplumber.open(file) as pdf:
                                    page = pdf.pages[0]
                                    uploaded = st.write(page.extract_text())
                                    message = page.extract_text().encode("utf-8")
                                    file.name = None
                            except:
                                st.write("None")

                        elif file.name[-4:] == "docx":
                            raw_text = docx2txt.process(file)
                            uploaded = st.write(raw_text)
                            message = uploaded.encode("utf-8")
                            file.name = None

                        elif file.name[-3:] == "txt":
                            st.text(str(file.read(), "utf-8"))  # empty
                            raw_text = str(
                                file.read(), "utf-8"
                            )  # works with st.text and st.write,used for futher processing
                            # st.text(raw_text) # Works
                            uploaded = st.write(raw_text)
                            message = uploaded.encode("utf-8")
                            file.name = None

                        if st.checkbox("Do you want to keep Password?"):
                            passw = st.text_input("Password", type="password")
                            confirm_passw = st.text_input(
                                "Confirm Password", type="password"
                            )

                            if passw == "":
                                st.info("Enter Password")
                            elif passw == confirm_passw:
                                if st.success("Password Confirmed"):
                                    if st.button(
                                        "Click to Perform Steganography on the Files Provided"
                                    ):
                                        bits = 2
                                        st.text(
                                            "Inserting the Parasitic File into the Host file"
                                        )
                                        # confirm_passw_ = cp.shrecode_steg_e(confirm_passw)
                                        # print(confirm_passw_)
                                        # FileDownloader(confirm_passw_).download()
                                        host.insert_message(
                                            message, bits, file.name, confirm_passw
                                        )
                                        st.text("Saving the File")
                                        st.write(
                                            "Click on the Link to Download the Encrypted File"
                                        )
                                        host.save()

                            else:
                                st.warning("Password is not the same")

                        else:
                            passw = None
                            if st.button(
                                "Click to Perform Steganography on the Files Provided"
                            ):
                                bits = 2
                                st.text(
                                    "Inserting the Parasitic File into the Host file"
                                )
                                host.insert_message(message, bits, file.name, passw)
                                st.text("Saving the File")
                                st.write(
                                    "Click on the Link to Download the Encrypted File"
                                )
                                host.save()

                elif choice1 == "Text":

                    uploaded_file = st.text_input("Enter the Text")
                    message = uploaded_file.encode("utf-8")
                    filename = None

                    if st.checkbox("Do you want to keep Password?"):
                        passw = st.text_input("Password", type="password")
                        confirm_passw = st.text_input(
                            "Confirm Password", type="password"
                        )

                        if passw == "":
                            st.info("Enter Password")

                        elif passw == confirm_passw:
                            st.success("Password Confirmed")
                            if st.button(
                                "Click to Perform Steganography on the Files Provided"
                            ):
                                bits = 2
                                st.text(
                                    "Inserting the Parasitic File into the Host file"
                                )
                                # confirm_passw_ = cp.shrecode_steg_e(confirm_passw)
                                # print(confirm_passw_)
                                # confirm_passw = cp.shrecode_steg_d(confirm_passw_)
                                # print(confirm_passw)
                                # FileDownloader(confirm_passw_).download()
                                host.insert_message(
                                    message, bits, filename, confirm_passw
                                )
                                st.text("Saving the File")
                                st.write(
                                    "Click on the Link to Download the Encrypted File"
                                )
                                host.save()

                        else:
                            st.warning("Password is not the same")

                    else:
                        passw = None
                        if st.button(
                            "Click to Perform Steganography on the Files Provided"
                        ):
                            bits = 2
                            st.text("Inserting the Parasitic File into the Host file")
                            host.insert_message(message, bits, filename, passw)
                            st.text("Saving the File")
                            st.write("Click on the Link to Download the Encrypted File")
                            host.save()

        else:
            st.text("Upload the Stego File which you want to decode.")
            x = st.file_uploader(
                "Upload the Host file", type=["png", "gif", "wav", "jpg"]
            )

            if x is not None:
                file_details = {
                    "Filename": x.name,
                    "FileType": x.type,
                    "FileSize": x.size,
                }
                st.write(file_details)
                with open(os.path.join("tempDir", "stego_file" + x.name), "wb") as f:
                    f.write(x.getbuffer())

                host_path = "tempDir/stego_file" + x.name
                host = HostElement(host_path)

                if st.checkbox("Does it have a Password?"):
                    passwo = st.text_input("Enter the Password", type="password")
                    if passwo != "":
                        if st.button(
                            "Click to Perform Decryption on the Files Provided"
                        ):
                            bits = 2
                            # co_pass = cp.shrecode_steg_e(passwo)
                            # passwo = cp.shrecode_steg_d(co_pass)
                            # print(passwo)
                            host.read_message(passwo)

                else:
                    passwo = None
                    if st.button("Click to Perform Decryption on the Files Provided"):
                        host.read_message(passwo)

    else:
        st.subheader("Cryptography")
        choice1 = st.selectbox("Crpytography types", crypt)
        if choice1 == "HYBRID":
            st.text("HYBRID")
            op = st.selectbox("Operation Type", ch)
            if op == "Encode":
                y = st.text_input("Enter the Text you want to Encode")
                x = st.number_input("Enter the Number(>=1024) for the generation of Keypair", 1023, 3072)
                if x >= 1024:
                    keyPair = RSA.generate(x)
                    if st.button("Encrypt"):
                        message = base64_cry_e(y)
                        encrypted  = encrypt__(message, keyPair)
                        st.text("Encrpted Message")
                        st.success(encrypted)

            else:
                y = st.text_input("Enter the Text you want to Decode")
                x = st.number_input("Enter the Number(>=1024) for the generation of Keypair", 1023, 3072)
                if x >= 1024:
                    keyPair = RSA.generate(x)
                    if st.button("Decrypt"):
                        decrypted = decrypt__(keyPair, y)
                        st.text("Decrpted Message")
                        st.success(decrypted)

        elif choice1 == "SELF":
            st.text("SELF")
            op = st.selectbox("Operation Type", ch)
            if op == "Encode":
                st.text("SELF Encryption")
                message = st.text_input("Enter the Text you want to Encode")
                if st.button("Encrypt"):
                    cp.shrecode_e(message)

            else:
                st.text("SELF Decryption")
                message = st.text_input("Enter the Text you want to Decode")
                if st.button("Decrypt"):
                    cp.shrecode_d(message)

        elif choice1 == "Hashes":
            st.subheader("Hashes")
            hash = st.selectbox("Select the Hashes Type", hashes)
            if hash == "MD5":
                st.subheader("MD5")
                message = st.text_input("Enter the Text")
                if st.button("Encrypt"):
                    cp.md5_e(message)

            elif hash == "SHA1":
                st.subheader("SHA1")
                message = st.text_input("Enter the Text")
                if st.button("Encrypt"):
                    cp.sha1_e(message)

            elif hash == "SHA256":
                st.subheader("SHA256")
                message = st.text_input("Enter the Text")
                if st.button("Encrypt"):
                    cp.sha256_e(message)

            elif hash == "SHA224":
                st.subheader("SHA224")
                message = st.text_input("Enter the Text")
                if st.button("Encrypt"):
                    cp.sha224_e(message)

            elif hash == "SHA512":
                st.subheader("SHA512")
                message = st.text_input("Enter the Text")
                if st.button("Encrypt"):
                    cp.sha512_e(message)

        elif choice1 == "Base64":
            st.subheader("BASE64")
            op = st.selectbox("Operation Type", ch)
            if op == "Encode":
                st.text("Base64 Encryption")
                message = st.text_input("Enter the Text you want to Encode")
                if st.button("Encrypt"):
                    cp.base64_e(message)

            else:
                st.text("Base64 Decryption")
                message = st.text_input("Enter the Text you want to Decode")
                if st.button("Decrypt"):
                    cp.base64_d(message)

        elif choice1 == "FPE":
            st.subheader("FPE")
            fpetype = st.selectbox("FPE Types", fp)
            if fpetype == "Numeric":
                st.subheader("FPE (Numeric)")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("FPE Numeric Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    if st.button("Encrypt"):
                        cp.fpenum_e(message)
                else:
                    st.text("FPE Numeric Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    if st.button("Decrypt"):
                        cp.fpenum_d(message)

            else:
                st.subheader("FPE (Alhabetic)")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("FPE Alpabetic Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    if st.button("Encrypt"):
                        cp.fpealp_e(message)

                else:
                    st.text("FPE Alpabetic Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    if st.button("Decrypt"):
                        cp.fpealp_d(message)

        elif choice1 == "Ciphers":
            st.subheader("Ciphers")
            cip = st.selectbox("Cipher Types", cipher)
            if cip == "Hill Cipher":
                st.subheader("Hill Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Hill Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    n = st.slider("What will be the order of square matrix: ", 2, 20, 1)
                    s = st.text_input("Enter the key", type="password")
                    if st.button("Encrypt"):
                        cp.hill_e(message, n, s)

                else:
                    st.text("Hill Cipher Decryption")
                    message = st.text_input("Enter your Text you want to Decode")
                    n = st.slider("What will be the order of square matrix: ", 2, 20, 1)
                    s = st.text_input("Enter the key", type="password")
                    if st.button("Decrypt"):
                        cp.hill_d(message, n, s)

            elif cip == "Playfair cipher":
                st.subheader("Playfair Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Playfair Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    key = st.text_input("Enter the key", type="password")
                    if st.button("Encrypt"):
                        cp.playfair_e(message, key)

                else:
                    st.text("Playfair Cipher Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    key = st.text_input("Enter the key", type="password")
                    if st.button("Decrypt"):
                        cp.playfair_d(message, key)

            elif cip == "Rail-fence cipher":
                st.subheader("Railfence Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Railfence Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    n = st.slider("Enter the number of rails: ", 1, 20, 1)
                    if st.button("Encrypt"):
                        cp.railfence_e(message, n)

                else:
                    st.text("Railfence Cipher Decryption")
                    message = st.text_input("Enter your Text you want to Decode")
                    n = st.slider("Enter the number of rails: ", 1, 20, 1)
                    if st.button("Decrypt"):
                        cp.railfence_d(message, n)

            elif cip == "Vernam cipher":
                st.subheader("Vernam Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Vernam Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    key = st.text_input(
                        "Enter the one time password: ", type="password"
                    )
                    if st.button("Encrypt"):
                        cp.vernam_e(message, key)

                else:
                    st.text("Vernam Cipher Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    key1 = st.text_input(
                        "Enter your one time password: ", type="password"
                    )
                    if st.button("Decrypt"):
                        cp.vernam_d(message, key1)

            elif cip == "Vigenere cipher":
                st.subheader("Vigenere Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Vigenere Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    key1 = st.text_input(
                        "Enter the one time password: ", type="password"
                    )
                    if st.button("Encrypt"):
                        cp.vigenere_e(message, key1)

                else:
                    st.text("Vigenere Cipher Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    key1 = st.text_input(
                        "Enter your one time password: ", type="password"
                    )
                    if st.button("Decrypt"):
                        cp.vigenere_d(message, key1)

            elif cip == "Caesar cipher":
                st.subheader("Caesar Cipher")
                op = st.selectbox("Operation Type", ch)
                if op == "Encode":
                    st.text("Caesar Cipher Encryption")
                    message = st.text_input("Enter the Text you want to Encode")
                    key1 = st.text_input(
                        "Enter the one time password: ", type="password"
                    )
                    if st.button("Encrypt"):
                        cp.caesar_e(message, key1)

                else:
                    st.text("Caesar Cipher Decryption")
                    message = st.text_input("Enter the Text you want to Decode")
                    key1 = st.text_input(
                        "Enter the one time password: ", type="password"
                    )
                    if st.button("Decrypt"):
                        cp.caesar_d(message, key1)

        else:
            st.info(
                """
            >HASHES

   ~MD5

    The MD5 message-digest algorithm is a hash function of 128-bit hash value.
    
    >Digest size: 128 bit

    >Block size:512 bit

   ~SHA1

    SHA-1 produces a 160-bit hash value known as a message digest aka hexadecimal number, 40 digits long.
    
    >Digest size: 160 bit

    >Block size: 512 bit

   ~SHA2

    SHA-2 is built using the Merkle–Damgård 
    construction, from a one-way compression 
    function itself built 
    using the Davies–Meyer structure
    from a specialized block cipher.

    common SHA-2 digests: 

        >~SHA224 

        >~SHA512

        >~SHA256

~FORMAT PRESERVING ENCRYPTION

   ~Format-preserving encryption (FPE), refers to encrypting in such a way that the output (the ciphertext) is in the same format as the input (the plaintext). The meaning of "format" varies.

    Example:

    ~encrypting a 6-digit password and the output cipher be 6-digit

    ~encrypting an English word and the output cipher be an english word

    ~encrypting a n-bit number and the ciphertext be n-bit.

>BASE64

   Base64 is a group of binary-to-text encoding schemes that represent binary data (more specifically a sequence of 8-bit bytes) in an ASCII string format by translating it into a radix-64 representation.

    >For more info on the Algorithm:

    >https://www.lucidchart.com/techblog/2017/10/23/base64-encoding-a-visual-explanation/


~CIPHERS

_____________________________________________________________________________________________________________________

   ~HILL CIPHER

    Hill cipher is a polygraphic substitution
    cipher based on linear algebra.

    >For more info on the Algorithm:

    >https://massey.limfinity.com/207/hillcipher.pdf

   ~PLAYFAIR

    Playfair cipher or Playfair square or Wheatstone–Playfair 
    cipher is a manual symmetric encryption technique and was 
    the first literal digram substitution cipher.

    >For more info on the Algorithm:

    >https://en.wikipedia.org/wiki/Playfair_cipher

   ~RAIL FENCE

    The rail fence cipher (also called a zigzag cipher) is a 
    form of transposition cipher. It 
    derives its name from the way in which it is encoded. 

    >For more info on the Algorithm:

    >http://rumkin.com/tools/cipher/railfence.php

    ~VERNAM

    Vernam cipher is a symmetrical stream cipher in which 
    the plaintext is combined with a random or pseudorandom 
    stream of data (the "keystream") of the same length, to 
    generate the ciphertext, using the Boolean "exclusive or" 
    XOR) function.

    >For more info on the Algorithm:

    >https://en.wikipedia.org/wiki/Gilbert_Vernam

   ~VIGENERE

    The Vigenère cipher is a method of encrypting alphabetic
    text by using a series of interwoven Caesar ciphers, 
    based on the letters of a 
    keyword. It employs a form of polyalphabetic substitution

    >For more info on the Algorithm:

    >https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher#Cryptanalysis
 
   ~CAESAR

    Caesar's cipher is one of the simplest and most 
    widely known encryption techniques. It is a type 
    of substitution cipher in which each letter in the 
    plaintext is replaced by a letter some fixed number 
    of positions down the alphabet. For example, with a 
    left shift of 3, D would be replaced by A, E would 
    become B, and so on. The method is named after Julius 
    Caesar, who used it in his private correspondence.

    >For more info on the Algorithm:

    >https://en.wikipedia.org/wiki/Caesar_cipher#Breaking_the_cipher

~SELF

    ~SELF is a modified type of the Shift Cipher. SELF can 
    encrypt alphanumeric (+symbols) unlike other ciphers ¯\_(ツ)_/¯
            """
            )


if __name__ == "__main__":
    main()
