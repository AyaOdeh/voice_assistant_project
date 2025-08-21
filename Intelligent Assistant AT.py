import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import requests
import json
import re
from gtts import gTTS
import pygame
import io
import threading
import time
import numpy as np
import pyaudio
import subprocess
import sys
import random
from fuzzywuzzy import process, fuzz

# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLU import *

# --- Global State Variables for GUI and Audio Visualization ---
assistant_state = {'status': "جاهز...", 'command': "...", 'log': "سجل المساعد:\n"}
state_lock = threading.Lock() # Lock to protect assistant_state updates

freq_energies = {'bass': 0.0, 'mid': 0.0, 'treble': 0.0}
audio_data_lock = threading.Lock() # Lock to protect freq_energies updates

running_pygame_gui = True # Flag to control the Pygame main loop

# --- Pygame GUI and OpenGL 3D Visualization Settings ---
pygame.init()
WIDTH, HEIGHT = 400, 400
# Set display mode to OPENGL and DOUBLEBUF for smooth 3D rendering
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME)
pygame.display.set_caption("AT: المساعد الصوتي المرئي ثلاثي الأبعاد")

# 3D Shapes Settings
BASE_RADIUS = 0.8 # Reverted to BASE_RADIUS for spheres
MAX_SCALE_BOOST = 0.5
MAX_MOVEMENT_OFFSET = 0.15
MOVEMENT_SPEED = 1.0

# --- PyAudio Setup ---
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
P = pyaudio.PyAudio()
mic_stream = None

# --- Audio Monitoring and FFT Analysis Thread Function ---
def audio_monitor_thread():
    global freq_energies, mic_stream

    FREQ_BANDS = {
        'bass': (20, 250),
        'mid': (250, 2000),
        'treble': (2000, 10000)
    }

    freqs = np.fft.fftfreq(CHUNK, d=1/RATE)
    band_indices = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        indices = np.where((np.abs(freqs) >= low) & (np.abs(freqs) <= high))[0]
        band_indices[band_name] = indices

    try:
        input_device_index = -1
        try:
            default_input_device_info = P.get_default_input_device_info()
            input_device_index = default_input_device_info['index']
            print(f"باستخدام جهاز الإدخال الافتراضي: {default_input_device_info.get('name')}")
        except Exception as e:
            print(f"تحذير: لم يتم العثور على جهاز إدخال صوت افتراضي. {e}")
            print("جاري البحث عن أي جهاز إدخال متاح...")
            info = P.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (P.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    input_device_index = i
                    print(f"باستخدام جهاز الإدخال: {P.get_device_info_by_host_api_device_index(0, i).get('name')}")
                    break
            if input_device_index == -1:
                print("خطأ: لم يتم العثور على أي جهاز إدخال صوت. لن يتمكن المساعد من الاستماع أو عرض الترددات.")
                global running_pygame_gui
                running_pygame_gui = False
                return

        mic_stream = P.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=input_device_index)

        print("بدء مراقبة الميكروفون لتحليل الترددات...")
        while running_pygame_gui:
            try:
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)

                yf = np.fft.fft(audio_np)
                magnitude_spectrum = np.abs(yf[:CHUNK // 2])

                current_energies = {}
                for band_name, indices in band_indices.items():
                    valid_indices = indices[indices < len(magnitude_spectrum)]
                    if len(valid_indices) > 0:
                        energy = np.sum(magnitude_spectrum[valid_indices]) / len(valid_indices)
                        current_energies[band_name] = energy
                    else:
                        current_energies[band_name] = 0.0

                MAX_ENERGY_BASS = 300000
                MAX_ENERGY_MID = 500000
                MAX_ENERGY_TREBLE = 250000

                with audio_data_lock:
                    freq_energies['bass'] = min(current_energies.get('bass', 0) / MAX_ENERGY_BASS, 1.0)
                    freq_energies['mid'] = min(current_energies.get('mid', 0) / MAX_ENERGY_MID, 1.0)
                    freq_energies['treble'] = min(current_energies.get('treble', 0) / MAX_ENERGY_TREBLE, 1.0)

            except IOError as e:
                if e.errno == -9988:
                    pass
                else:
                    print(f"خطأ في PyAudio (مراقبة الصوت): {e}")
                    raise e
            time.sleep(0.01)
    except Exception as e:
        print(f"خطأ حرج في مراقبة الصوت: {e}")
    finally:
        if mic_stream:
            mic_stream.stop_stream()
            mic_stream.close()
        P.terminate()
        print("إيقاف مراقبة الميكروفون.")

# --- Arabic Voice Assistant Class ---
class ArabicVoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.setup_voice()
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        pygame.mixer.init()
        
        self.user_dirs = {
            'سطح المكتب': os.path.join(os.path.expanduser('~'), 'Desktop'),
            'المستندات': os.path.join(os.path.expanduser('~'), 'Documents'),
            'التنزيلات': os.path.join(os.path.expanduser('~'), 'Downloads'),
            'الموسيقى': os.path.join(os.path.expanduser('~'), 'Music'),
            'الصور': os.path.join(os.path.expanduser('~'), 'Pictures'),
            'الفيديوهات': os.path.join(os.path.expanduser('~'), 'Videos'),
            'البرامج': os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs'),
            'ملفات البرنامج x86': os.path.join('C:', 'Program Files (x86)') if os.name == 'nt' else None,
            'ملفات البرنامج': os.path.join('C:', 'Program Files') if os.name == 'nt' else None
        }
        self.user_dirs = {k: v for k, v in self.user_dirs.items() if v is not None and os.path.exists(v)}

        self.app_knowledge_base = {
            'chrome': ['متصفح', 'جوجل كروم', 'افتح ويب'],
            'firefox': ['فايرفوكس'],
            'notepad': ['مفكرة', 'نوت باد', 'برنامج كتابة'],
            'calc': ['آلة حاسبة', 'كالكيوليتر'],
            'cmd': ['موجه الأوامر', 'سي ام دي', 'الطرفية'],
            'explorer': ['ملفاتي', 'مستكشف الملفات', 'الملفات'],
            'wmplayer': ['مشغل الموسيقى', 'مشغل الفيديو', 'ويندوز ميديا بلاير', 'موسيقى'],
            'control': ['لوحة التحكم'],
            'mspaint': ['الرسام'],
            'excel': ['اكسل', 'جداول البيانات', 'برنامج حسابات'],
            'winword': ['وورد', 'مايكروسوفت وورد', 'محرر نصوص'],
            'powerpnt': ['باوربوينت', 'عروض تقديمية', 'برزنتيشن'],
            'outlook': ['أوتلوك', 'بريد إلكتروني'],
            'code': ['في اس كود', 'فجوال ستوديو كود', 'المبرمج', 'محرر الأكواد'],
            'vlc': ['في ال سي', 'مشغل وسائط في ال سي'],
            'steam': ['ستيم', 'الألعاب'],
            'discord': ['ديسكورد'],
            'zoom': ['زووم', 'اجتماعات'],
            'skype': ['سكايب'],
            'photoshop': ['فوتوشوب', 'محرر صور', 'برنامج صور'],
            'illustrator': ['الستريتور'],
            'premiere': ['بريمير برو', 'محرر فيديو'],
            'autocad': ['اوتوكاد'],
            'maya': ['مايا'],
            'blender': ['blender'],
            'ms-settings:': ['الإعدادات', 'اعدادات الجهاز', 'ضبط'],
            'microsoft-edge:///?ux=copilot&tcp=1&source=taskbar': ['كوبايلوت', 'المساعد الذكي', 'مايكروسوفت كوبايلوت']
        }
        self.reverse_app_map = {}
        for cmd, aliases in self.app_knowledge_base.items():
            for alias in aliases:
                self.reverse_app_map[alias.lower()] = cmd
            self.reverse_app_map[cmd.lower()] = cmd

        self.commands = {
            'وقت': self.get_time,
            'تاريخ': self.get_date,
            'طقس': self.get_weather,
            'بحث': self.search_web,
            'افتح': self.open_application,
            'موسيقى': self.play_music,
            'توقف': self.stop_assistant,
            'اغلاق': self.stop_assistant,
            'مساعدة': self.show_help,
            'إيقاف تشغيل': self.shutdown_system,
            'صباح الخير': self.greet_user,
            'مساء الخير': self.greet_user,
            'نكتة': self.tell_joke,
            'شكراً': self.say_thank_you,
            'شكرا': self.say_thank_you,
        }
        
        self.running = True
        
        self.jokes = [
            "لماذا لا تثق الذرة أبداً؟ لأنها تتكون من عناصر لا يمكن الوثوق بها!",
            "ما هو الحيوان الذي إذا حذفنا أول حرف منه أصبح شيئاً نشربه؟ زرافة - رافعة - رافع.",
            "ماذا قال الجدار للجدار؟ سأقابلك عند الزاوية!",
            "ما الفرق بين النملة والفيل؟ الفيل لديه أرجل، أما النملة فلديها نمول!",
            "لماذا وضعوا كاميرا في الغرفة؟ عشان يراقبوا الغرفة!"
        ]
        
        # --- AI Context and Preferences ---
        self.last_known_intent = None
        self.last_known_entity = {}
        self.user_preferences = {
            'preferred_weather_city': "الرياض"
        }
        self.weather_city_ask_count = {}

    def setup_voice(self):
        voices = self.engine.getProperty('voices')
        found_arabic_voice = False
        for voice in voices:
            if 'arabic' in voice.name.lower() or 'ar' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                found_arabic_voice = True
                break
        if not found_arabic_voice:
            print("لم يتم العثور على صوت عربي في pyttsx3. سيتم استخدام الصوت الافتراضي.")
        
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.8)
    
    def speak_arabic(self, text):
        with state_lock:
            assistant_state['status'] = "أتحدث..."
            assistant_state['command'] = text
        
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"خطأ في النطق باستخدام gTTS: {e}. سأستخدم pyttsx3.")
            self.engine.say(text)
            self.engine.runAndWait()
        
        with state_lock:
            assistant_state['status'] = "جاهز..."
            assistant_state['command'] = "..."
            assistant_state['log'] += f"المساعد: {text}\n"

    def listen(self, timeout=5, phrase_time_limit=7):
        with state_lock:
            assistant_state['status'] = "أستمع..."
            assistant_state['command'] = "..."
            assistant_state['log'] += "المساعد: أستمع إليك...\n"

        try:
            with self.microphone as source:
                print("🎤 أستمع إليك...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5) 
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("🔄 معالجة الصوت...")
            with state_lock:
                assistant_state['status'] = "أعالج الصوت..."
            
            try:
                command = self.recognizer.recognize_google(audio, language="ar-SA")
                print(f"تم سماع: {command}")
                with state_lock:
                    assistant_state['command'] = f"أنت: {command}"
                    assistant_state['log'] += f"أنت: {command}\n"
                return command.lower()
            except sr.UnknownValueError:
                print("لم أتمكن من فهم ما قلته (محاولة عربية، تجربة إنجليزية...)")
                try:
                    command = self.recognizer.recognize_google(audio, language="en-US")
                    print(f"English command: {command}")
                    with state_lock:
                        assistant_state['command'] = f"أنت (إنجليزية): {command}"
                        assistant_state['log'] += f"أنت (إنجليزية): {command}\n"
                    return command.lower()
                except sr.UnknownValueError:
                    print("لم أتمكن من فهم ما قلته")
                    with state_lock:
                        assistant_state['log'] += "المساعد: لم أتمكن من فهم ما قلته.\n"
                    return ""
            
        except sr.WaitTimeoutError:
            print("لم أسمع شيئاً...")
            with state_lock:
                assistant_state['log'] += "المساعد: لم أسمع شيئاً.\n"
            return ""
        except sr.UnknownValueError:
            print("لم أتمكن من فهم ما قلته")
            with state_lock:
                assistant_state['log'] += "المساعد: لم أتمكن من فهم ما قلته.\n"
            return ""
        except sr.RequestError as e:
            print(f"خطأ في الخدمة: {e}")
            with state_lock:
                assistant_state['log'] += f"المساعد: خطأ في الخدمة: {e}\n"
            return ""
        finally:
            with state_lock:
                assistant_state['status'] = "جاهز..."

    def process_command(self, command):
        if not command:
            self.last_known_intent = None
            self.last_known_entity = {}
            return
        
        command_keywords = list(self.commands.keys())
        
        best_match_keyword, score = process.extractOne(command, command_keywords, scorer=fuzz.ratio)
        
        MATCH_THRESHOLD = 70 

        if score >= MATCH_THRESHOLD:
            print(f"Matched command: '{best_match_keyword}' with score: {score}")
            self.last_known_intent = best_match_keyword
            if best_match_keyword == 'طقس':
                city = self._extract_city_from_command(command)
                if city:
                    self.last_known_entity['city'] = city
                    self.weather_city_ask_count[city] = self.weather_city_ask_count.get(city, 0) + 1
                    if self.weather_city_ask_count[city] >= 2:
                        self.user_preferences['preferred_weather_city'] = city
                        print(f"تم تعلم المدينة المفضلة للطقس: {city}")
                else:
                    self.last_known_entity['city'] = None

            self.commands[best_match_keyword](command)
        else:
            print(f"No direct command match (score: {score}). Attempting contextual understanding for: {command}")
            if self.last_known_intent == 'طقس' and not self.last_known_entity.get('city'):
                city = self._extract_city_from_command(command)
                if city:
                    print(f"Using context: Assuming weather query for {city}")
                    self.last_known_entity['city'] = city
                    self.commands['طقس'](f"طقس في {city}")
                    return
            
            self.last_known_intent = None
            self.last_known_entity = {}
            self.handle_general_query(command)
        
        if best_match_keyword not in ['طقس', 'بحث']:
             self.last_known_intent = None
             self.last_known_entity = {}

    def _extract_city_from_command(self, command):
        city = None
        match = re.search(r'(?:في|بـ|عن)\s+([ا-ي\s]+)', command)
        if match:
            city = match.group(1).strip()
        elif "طقس" in command and len(command.split()) > 1:
            words = command.split()
            if "طقس" in words:
                index = words.index("طقس")
                if index + 1 < len(words):
                    city = words[index + 1].strip()
        return city
        
    def get_time(self, command=""):
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M")
        response = f"الوقت الآن هو {time_str}"
        print(response)
        self.speak_arabic(response)
    
    def get_date(self, command=""):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y/%m/%d")
        response = f"تاريخ اليوم هو {date_str}"
        print(response)
        self.speak_arabic(response)
    
    def get_weather(self, command):
        try:
            API_KEY = "673bc8a696dc8231c1f3db5c2b6bd2d7" # Replace with your OpenWeatherMap API key if needed
            
            if not API_KEY:
                response = "يرجى إضافة API key للطقس من OpenWeatherMap.org."
                print(response)
                self.speak_arabic(response)
                return
            
            city = self._extract_city_from_command(command)
            
            if not city:
                if self.last_known_entity.get('city'):
                    city = self.last_known_entity['city']
                    print(f"استخدام المدينة من السياق: {city}")
                elif self.user_preferences.get('preferred_weather_city'):
                    city = self.user_preferences['preferred_weather_city']
                    print(f"استخدام المدينة المفضلة: {city}")
                else:
                    city = "الرياض" # Fallback default
                    print(f"استخدام المدينة الافتراضية: {city}")

            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=ar"
            
            response_data = requests.get(url)
            
            if response_data.status_code == 200:
                data = response_data.json()
                
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                
                response = f"الطقس في {city}: {description}، درجة الحرارة {temp:.1f} درجة مئوية، تشعر وكأنها {feels_like:.1f} درجة، الرطوبة {humidity}%"
                
            else:
                response = f"لم أتمكن من العثور على معلومات الطقس لمدينة {city}. رمز الخطأ: {response_data.status_code}"
            
            print(response)
            self.speak_arabic(response)
            
        except Exception as e:
            response = "حدث خطأ في الحصول على معلومات الطقس. تأكد من اتصالك بالإنترنت ومفتاح الـ API."
            print(f"خطأ: {e}")
            print(response)
            self.speak_arabic(response)
    
    def search_web(self, command):
        try:
            search_term = command.replace("بحث", "").replace("عن", "").strip()
            if search_term:
                url = f"https://www.google.com/search?q={search_term}"
                webbrowser.open(url)
                response = f"تم البحث عن {search_term}"
            else:
                response = "ماذا تريد أن تبحث عنه؟"
            
            print(response)
            self.speak_arabic(response)
            
        except Exception as e:
            response = "حدث خطأ في البحث على الويب."
            print(response)
            self.speak_arabic(response)
    
    def open_application(self, command):
        app_or_file_name_raw = command.replace("فتح", "").strip()
        if not app_or_file_name_raw:
            self.speak_arabic("ما هو التطبيق أو الملف الذي تريد فتحه؟")
            return

        target_to_open = None
        
        words_in_command = app_or_file_name_raw.split()
        potential_app_name = " ".join([word for word in words_in_command if word not in ["برنامج", "تطبيق", "ملف", "افتح"]])
        potential_app_name = potential_app_name.lower()

        best_match_key, score = process.extractOne(potential_app_name, self.reverse_app_map.keys(), scorer=fuzz.ratio)
        
        if score < 70 and fuzz.partial_ratio(potential_app_name, best_match_key) >= 90:
             score = fuzz.partial_ratio(potential_app_name, best_match_key)

        if score >= 75:
            cmd_to_execute = self.reverse_app_map[best_match_key]
            print(f"AI-Match found: '{potential_app_name}' matched with '{best_match_key}' -> execute '{cmd_to_execute}' (score: {score})")
            target_to_open = cmd_to_execute
        
        match_dir_file = re.search(r'(.+)\s+في\s+(.+)', app_or_file_name_raw)
        if match_dir_file:
            file_name_search = match_dir_file.group(1).strip()
            dir_name_spoken = match_dir_file.group(2).strip()
            
            target_dir_path = None
            for key, path in self.user_dirs.items():
                if key.lower() in dir_name_spoken.lower():
                    target_dir_path = path
                    break
            
            if target_dir_path:
                found_item_path = None
                for root, dirs, files in os.walk(target_dir_path):
                    for name in files:
                        if fuzz.partial_ratio(file_name_search.lower(), name.lower()) >= 80:
                            found_item_path = os.path.join(root, name)
                            break
                    if found_item_path:
                        break
                    for name in dirs:
                        if fuzz.partial_ratio(file_name_search.lower(), name.lower()) >= 80:
                            found_item_path = os.path.join(root, name)
                            break
                    if found_item_path:
                        break

                if found_item_path:
                    target_to_open = found_item_path
                else:
                    response = f"لم أجد '{file_name_search}' في مجلد {dir_name_spoken} أو مجلداته الفرعية."
                    print(response)
                    self.speak_arabic(response)
                    return
            else:
                response = f"لم أتمكن من التعرف على المجلد {dir_name_spoken}. جرب اسماً شائعاً مثل سطح المكتب أو المستندات."
                print(response)
                self.speak_arabic(response)
                return

        if not target_to_open:
            if os.path.exists(app_or_file_name_raw):
                target_to_open = app_or_file_name_raw
            else:
                target_to_open = app_or_file_name_raw
        
        if target_to_open:
            try:
                if target_to_open.startswith("ms-settings:") or target_to_open.startswith("microsoft-edge://"):
                    webbrowser.open(target_to_open)
                    response = f"تم فتح {app_or_file_name_raw}"
                elif target_to_open.startswith("http"):
                    webbrowser.open(target_to_open)
                    response = f"تم فتح {app_or_file_name_raw}"
                else:
                    if os.name == 'nt' and os.path.exists(target_to_open):
                         os.startfile(target_to_open)
                    else:
                        subprocess.Popen(target_to_open, shell=True)
                    response = f"تم فتح {os.path.basename(target_to_open)}"
                
                print(response)
                self.speak_arabic(response)
                return
            except FileNotFoundError:
                response = f"لم أجد التطبيق أو الملف '{os.path.basename(target_to_open)}'. تأكد من وجوده أو من تهيئته في متغيرات النظام."
                print(response)
                self.speak_arabic(response)
                return
            except Exception as e:
                response = f"حدث خطأ أثناء محاولة فتح {os.path.basename(target_to_open)}: {e}"
                print(response)
                self.speak_arabic(response)
                return
        
        response = f"لم أتمكن من تحديد التطبيق أو الملف المطلوب: '{app_or_file_name_raw}'. يرجى التأكد من الاسم أو المسار."
        print(response)
        self.speak_arabic(response)
    
    def play_music(self, command):
        try:
            self.open_application("افتح مشغل الموسيقى")
        except Exception as e:
            response = "لم أتمكن من تشغيل الموسيقى. تأكد من وجود مشغل وسائط افتراضي."
            print(response)
            self.speak_arabic(response)
    
    def shutdown_system(self, command=""):
        self.speak_arabic("هل أنت متأكد أنك تريد إيقاف تشغيل الجهاز؟ قل نعم للتأكيد أو لا للإلغاء.")
        confirmation = self.listen(timeout=7, phrase_time_limit=4)
        
        if "نعم" in confirmation or "أجل" in confirmation:
            self.speak_arabic("حسناً، جاري إيقاف تشغيل الجهاز.")
            print("جاري إيقاف تشغيل الجهاز...")
            os.system("shutdown /s /t 1")
        else:
            self.speak_arabic("تم إلغاء إيقاف التشغيل.")
            print("تم إلغاء إيقاف التشغيل.")
    
    def greet_user(self, command):
        current_hour = datetime.datetime.now().hour
        if "صباح الخير" in command:
            response = "صباح النور! كيف يمكنني مساعدتك؟"
        elif "مساء الخير" in command:
            response = "مساء النور! أتمنى لك أمسية سعيدة."
        elif 5 <= current_hour < 12:
            response = "صباح الخير! كيف يمكنني مساعدتك؟"
        elif 12 <= current_hour < 18:
            response = "مساء الخير! كيف يمكنني مساعدتك؟"
        else:
            response = "مساء الخير! أتمنى لك ليلة هادئة."
        
        self.speak_arabic(response)

    def tell_joke(self, command=""):
        joke = random.choice(self.jokes)
        self.speak_arabic(joke)
        print(f"نكتة: {joke}")

    def say_thank_you(self, command=""):
        responses = [
            "العفو! يسعدني مساعدتك.",
            "لا شكر على واجب.",
            "بكل سرور!",
            "أنا هنا لخدمتك."
        ]
        response = random.choice(responses)
        self.speak_arabic(response)
        print(f"استجابة للشكر: {response}")
    
    def ask_llm(self, prompt_text):
        print(f"Asking LLM: {prompt_text}")
        try:
            chat_history = []
            chat_history.append({"role": "user", "parts": [{"text": prompt_text}]})
            payload = {"contents": chat_history}
            api_key = "" # !! AQUIRE YOUR OWN GOOGLE AI STUDIO API KEY AND PASTE IT HERE !!
                         # !! IF RUNNING LOCALLY, OTHERWISE LEAVE IT BLANK FOR CANVAS ENV !!
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                text = result['candidates'][0]['content']['parts'][0].text
                print(f"LLM Response: {text}")
                return text
            else:
                print("LLM Response: No text content found in response.")
                return ""
        except requests.exceptions.RequestException as e:
            print(f"LLM Request Error: {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"LLM JSON Decode Error: {e}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred while asking LLM: {e}")
            return ""

    def handle_general_query(self, command):
        """
        Handles general or misunderstood queries.
        Tries LLM first, then falls back to web search.
        """
        print(f"معالجة استفسار عام: {command}")
        
        llm_response = self.ask_llm(command)
        if llm_response:
            response = f"حسب فهمي: {llm_response}"
            print(response)
            self.speak_arabic(response)
            return
        else:
            print("لم أجد إجابة من LLM. جاري البحث على الويب.")
        
        try:
            url = f"https://www.google.com/search?q={command}"
            webbrowser.open(url)
            response = f"لم أجد إجابة مباشرة، لكنني فتحت لك بحثاً على جوجل عن {command}."
            print(response)
            self.speak_arabic(response)
        except Exception as e:
            response = "أعتذر، حدث خطأ أثناء محاولة البحث على الويب."
            print(f"خطأ في البحث على الويب: {e}")
            print(response)
            self.speak_arabic(response)
    
    def show_help(self, command=""):
        help_text = """
        الأوامر المتاحة:
        - 'وقت' للحصول على الوقت الحالي
        - 'تاريخ' للحصول على التاريخ
        - 'طقس [في مدينة]' لمعرفة حالة الطقس (يمكنك الآن فقط قول 'طقس' بعد سؤالك عن مدينة ليعطيك طقس نفس المدينة، وسيتذكر المساعد مدينتك المفضلة للطقس!)
        - 'بحث عن...' للبحث في الويب
        - 'افتح [اسم التطبيق/الملف]' لفتح التطبيقات الشائعة أو الملفات بالاسم.
          (مثال: 'افتح كروم', 'افتح مفكرة', 'افتح سي ام دي', 'افتح برنامج الكتابة', 'افتح برامج الاكسل', 'افتح الإعدادات', 'افتح كوبايلوت', 'افتح مستكشف الملفات', 'افتح لوحة التحكم', 'افتح في اس كود')
        - 'افتح [اسم الملف] في [اسم المجلد]' (مثل: 'افتح تقرير في المستندات').
          (يبحث داخل المجلد المحدد ومجلداته الفرعية بمطابقة ذكية).
        - 'موسيقى' لتشغيل مشغل الوسائط الافتراضي.
        - 'إيقاف تشغيل الجهاز' لإيقاف تشغيل الكمبيوتر (يتطلب تأكيداً صوتياً)
        - 'توقف' أو 'اغلاق' لإيقاف المساعد
        - 'صباح الخير' أو 'مساء الخير' للتحية
        - 'نكتة' لإلقاء نكتة
        - 'شكراً' للشكر
        - 'مساعدة' لعرض هذه القائمة
        """
        print(help_text)
        self.speak_arabic("يمكنني مساعدتك في الوقت والتاريخ والبحث وفتح التطبيقات وإيقاف تشغيل الجهاز. قل 'مساعدة' لعرض قائمة الأوامر.")
    
    def stop_assistant(self, command=""):
        response = "وداعاً! كان من دواعي سروري مساعدتك."
        print(response)
        self.speak_arabic(response)
        self.running = False
        global running_pygame_gui
        running_pygame_gui = False
    
    def start_listening_loop(self):
        print("🤖 المساعد الصوتي الذكي جاهز!")
        print("قل 'مساعدة' لمعرفة الأوامر المتاحة")
        self.speak_arabic("أهلاً بك! أنا إيه تي، مساعدك الصوتي الذكي. كيف يمكنني مساعدتك؟")
        
        while self.running:
            try:
                command = self.listen()
                if command:
                    self.process_command(command)
            except KeyboardInterrupt:
                print("\nتم إيقاف المساعد يدوياً.")
                self.running = False
                global running_pygame_gui
                running_pygame_gui = False
            except Exception as e:
                print(f"خطأ غير متوقع في حلقة الاستماع: {e}")
                time.sleep(2)

# --- OpenGL Initialization Function ---
def init_opengl(width, height):
    # Initial clear color (dark blue/purple for space-like background)
    glClearColor(0.05, 0.05, 0.1, 0.0) # Dark blue, almost black
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 2.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.1, 0.1, 0.15, 1.0))

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)

# --- Pygame GUI Main Loop Function ---
def run_visual_assistant_gui():
    global running_pygame_gui

    init_opengl(WIDTH, HEIGHT)

    monitor_thread = threading.Thread(target=audio_monitor_thread, daemon=True)
    monitor_thread.start()

    assistant = ArabicVoiceAssistant()
    logic_thread = threading.Thread(target=assistant.start_listening_loop, daemon=True)
    logic_thread.start()

    rotation_angle_y = 0.0
    rotation_angle_x = 0.0
    rotation_speed_y = 0.5
    rotation_speed_x = 0.3
    
    orb_offset_x = 0.0
    orb_offset_y = 0.0
    offset_speed = 0.5
    
    # Initialize quadric for gluSphere
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)

    while running_pygame_gui:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_pygame_gui = False
                assistant.stop_assistant()

        t = time.time()
        # Background color: Smooth gradient from dark blue/purple to dark cyan
        # Inspired by the image's background gradient
        bg_r = 0.05 + 0.05 * np.sin(t * 0.4 + np.pi/2) # Oscillate around 0.05
        bg_g = 0.05 + 0.05 * np.cos(t * 0.5) # Oscillate around 0.05
        bg_b = 0.1 + 0.08 * np.sin(t * 0.6) # Oscillate around 0.1
        glClearColor(bg_r, bg_g, bg_b, 0.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        with audio_data_lock:
            bass_energy = freq_energies['bass']
            mid_energy = freq_energies['mid']
            treble_energy = freq_energies['treble']

        overall_energy = (bass_energy + mid_energy + treble_energy) / 3

        # Main sphere color: Fixed vibrant purple/magenta from the image
        current_status_main_color = (0.8, 0.2, 0.8) # A vibrant purple/magenta

        # Glow color: Oscillate between magenta/pink and cyan/blue, reacting to audio
        # More vibrant and directly linked to the image's gradient
        glow_r_base = 0.9 # High red for pink/magenta
        glow_g_base = 0.1 # Low green
        glow_b_base = 0.9 # High blue for purple/magenta

        # Introduce a secondary color for the glow, leaning towards cyan/blue
        glow_r_alt = 0.1 # Low red for cyan/blue
        glow_g_alt = 0.9 # High green
        glow_b_alt = 0.9 # High blue

        # Blend between base and alternate glow colors based on overall energy and time
        blend_factor = (np.sin(t * 3.0) + 1) / 2 # Oscillates between 0 and 1
        
        glow_color_r = (glow_r_base * (1 - blend_factor) + glow_r_alt * blend_factor) * (0.8 + 0.2 * overall_energy)
        glow_color_g = (glow_g_base * (1 - blend_factor) + glow_g_alt * blend_factor) * (0.8 + 0.2 * overall_energy)
        glow_color_b = (glow_b_base * (1 - blend_factor) + glow_b_alt * blend_factor) * (0.8 + 0.2 * overall_energy)

        glow_color_r = min(1.0, max(0.0, glow_color_r))
        glow_color_g = min(1.0, max(0.0, glow_color_g))
        glow_color_b = min(1.0, max(0.0, glow_color_b))

        rotation_angle_y += rotation_speed_y
        rotation_angle_x += rotation_speed_x
        if rotation_angle_y > 360: rotation_angle_y -= 360
        if rotation_angle_x > 360: rotation_angle_x -= 360

        # --- Draw Main Sphere (fixed vibrant purple/magenta) ---
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)

        time_factor = time.time() * offset_speed
        orb_offset_x = np.sin(time_factor * 1.5) * MAX_MOVEMENT_OFFSET * overall_energy
        orb_offset_y = np.cos(time_factor * 1.2) * MAX_MOVEMENT_OFFSET * overall_energy
        glTranslatef(orb_offset_x, orb_offset_y, 0.0)

        glRotatef(rotation_angle_y, 0, 1, 0)
        glRotatef(rotation_angle_x, 1, 0, 0)

        scale_factor_main = 1.0 + overall_energy * MAX_SCALE_BOOST * 0.5
        glScalef(scale_factor_main, scale_factor_main, scale_factor_main)

        # Main sphere color is now a fixed vibrant purple/magenta
        glColor4f(current_status_main_color[0], current_status_main_color[1], current_status_main_color[2], 1.0)
        gluSphere(quadric, BASE_RADIUS, 32, 32)

        # Draw a slightly larger, semi-transparent sphere for the glow effect
        glPushMatrix()
        glScalef(1.1, 1.1, 1.1) # Slightly larger than the main sphere
        glColor4f(glow_color_r, glow_color_g, glow_color_b, 0.3) # Vibrant glow color with transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blending for glow
        gluSphere(quadric, BASE_RADIUS, 32, 32)
        glDisable(GL_BLEND)
        glPopMatrix()


        # --- Draw Second Sphere (reactive to bass energy, with vibrant glow) ---
        glPushMatrix()
        glTranslatef(np.sin(time_factor * 0.7) * 0.5, np.cos(time_factor * 0.9) * 0.5, np.sin(time_factor * 0.5) * 0.5)
        glRotatef(rotation_angle_y * 1.5, 0, 1, 0)
        glRotatef(rotation_angle_x * 0.8, 1, 0, 0)
        
        scale_factor_bass = 0.5 + bass_energy * MAX_SCALE_BOOST * 1.5
        glScalef(scale_factor_bass, scale_factor_bass, scale_factor_bass)

        # Second sphere color: A slightly different vibrant shade, perhaps more blue/cyan
        glColor4f(0.2, 0.8, 0.8, 1.0) # A vibrant cyan/blue
        gluSphere(quadric, BASE_RADIUS * 0.8, 32, 32)

        # Draw a glow for the second sphere, using the same dynamic glow colors
        glPushMatrix()
        glScalef(1.1, 1.1, 1.1) # Slightly larger
        glColor4f(glow_color_r, glow_color_g, glow_color_b, 0.4) # Semi-transparent glow
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blending
        gluSphere(quadric, BASE_RADIUS * 0.8, 32, 32)
        glDisable(GL_BLEND)
        glPopMatrix()

        glPopMatrix()

        pygame.display.flip()

        pygame.time.Clock().tick(60)

    gluDeleteQuadric(quadric) # Delete quadric when done
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_visual_assistant_gui()
