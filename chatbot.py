import openai
import json
import re
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class TravelChatbot:
    def __init__(self):
        # Khởi tạo OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Template prompt để trích xuất thông tin
        self.extraction_prompt = """
Bạn là một AI chuyên phân tích yêu cầu du lịch của người dùng.
Hãy phân tích câu hỏi sau và trích xuất thông tin về điều kiện thời tiết mong muốn.

Câu hỏi: "{user_input}"

QUAN TRỌNG: Hãy chú ý đặc biệt đến THÁNG được đề cập trong câu hỏi.

Hãy trả về kết quả dưới dạng JSON với các trường sau:
- avgtemp_c: nhiệt độ trung bình mong muốn (°C) - số thực từ 15-35
- maxwind_kph: tốc độ gió tối đa mong muốn (km/h) - số thực từ 5-30
- avghumidity: độ ẩm trung bình mong muốn (%) - số thực từ 50-90
- avgvis_km: tầm nhìn trung bình mong muốn (km) - số thực từ 5-15
- month: tháng du lịch (1-12) - PHẢI trích xuất chính xác từ câu hỏi, nếu không có thì null
- region: vùng miền mong muốn - nếu có đề cập, nếu không thì null
- terrain: địa hình mong muốn - nếu có đề cập, nếu không thì null
- preferences: mô tả ngắn gọn về sở thích du lịch

Quy tắc trích xuất chi tiết:

THÁNG:
- "tháng 1" hoặc "tháng một" -> 1
- "tháng 2" hoặc "tháng hai" -> 2
- "tháng 3" hoặc "tháng ba" -> 3
- "tháng 4" hoặc "tháng tư" -> 4
- "tháng 5" hoặc "tháng năm" -> 5
- "tháng 6" hoặc "tháng sáu" -> 6
- "tháng 7" hoặc "tháng bảy" -> 7
- "tháng 8" hoặc "tháng tám" -> 8
- "tháng 9" hoặc "tháng chín" -> 9
- "tháng 10" hoặc "tháng mười" -> 10
- "tháng 11" hoặc "tháng mười một" -> 11
- "tháng 12" hoặc "tháng mười hai" -> 12
- "mùa xuân" -> 2, "mùa hè" -> 6, "mùa thu" -> 9, "mùa đông" -> 12

NHIỆT ĐỘ (avgtemp_c):
- "20 độ", "20°C", "20 độ C" -> 20
- "mát mẻ", "se lạnh" -> 20
- "nóng", "ấm áp" -> 30
- "ôn hòa", "dễ chịu" -> 25
- "lạnh" -> 18

GIÓ (maxwind_kph):
- "10 km/h", "10km/h", "10 kmh" -> 10
- "gió nhẹ", "ít gió" -> 8
- "gió mạnh" -> 25
- "không thích gió mạnh" -> 10

ĐỘ ẨM (avghumidity):
- "60%", "60 phần trăm" -> 60
- "khô ráo", "khô" -> 55
- "ẩm ướt", "ẩm" -> 80
- "vừa phải" -> 70

TẦM NHÌN (avgvis_km):
- "12 km", "12km" -> 12
- "tầm nhìn xa", "tầm nhìn tốt" -> 12
- "tầm nhìn kém" -> 6
- "bình thường" -> 10

VÙNG MIỀN (region):
- "miền Bắc", "Bắc Bộ", "phía Bắc" -> "Trung du và miền núi Bắc Bộ" hoặc "Đồng bằng sông Hồng"
- "miền Nam", "Nam Bộ", "phía Nam" -> "Đồng bằng sông Cửu Long" hoặc "Đông Nam Bộ"
- "miền Trung", "Trung Bộ", "phía Trung" -> "Bắc Trung Bộ và Duyên hải miền Trung"
- "Tây Nguyên", "cao nguyên" -> "Tây Nguyên"
- "đồng bằng sông Hồng", "Hà Nội", "Hải Phòng" -> "Đồng bằng sông Hồng"
- "đồng bằng sông Cửu Long", "Mekong", "Cần Thơ", "An Giang" -> "Đồng bằng sông Cửu Long"

ĐỊA HÌNH (terrain):
- "miền núi", "núi", "vùng núi", "cao", "leo núi" -> "miền núi"
- "ven biển", "biển", "bãi biển", "tắm biển", "gần biển" -> "ven biển"
- "đồng bằng", "bằng phẳng", "đồng ruộng", "nông thôn" -> "đồng bằng"

Ví dụ:
- "Tôi muốn đi chơi vào tháng 11" -> {{"avgtemp_c": 25, "maxwind_kph": 15, "avghumidity": 70, "avgvis_km": 10, "month": 11, "region": null, "terrain": null, "preferences": "du lịch tháng 11"}}
- "Tôi muốn nơi mát mẻ 20 độ C vào tháng 12" -> {{"avgtemp_c": 20, "maxwind_kph": 15, "avghumidity": 70, "avgvis_km": 10, "month": 12, "region": null, "terrain": null, "preferences": "nơi mát mẻ 20°C tháng 12"}}
- "Du lịch biển miền Trung mùa hè" -> {{"avgtemp_c": 28, "maxwind_kph": 20, "avghumidity": 75, "avgvis_km": 12, "month": null, "region": "Bắc Trung Bộ và Duyên hải miền Trung", "terrain": "ven biển", "preferences": "du lịch biển miền Trung mùa hè"}}
- "Tôi thích leo núi ở Tây Nguyên" -> {{"avgtemp_c": 25, "maxwind_kph": 15, "avghumidity": 70, "avgvis_km": 10, "month": null, "region": "Tây Nguyên", "terrain": "miền núi", "preferences": "leo núi Tây Nguyên"}}
- "Nơi đồng bằng miền Nam, khô ráo" -> {{"avgtemp_c": 25, "maxwind_kph": 15, "avghumidity": 55, "avgvis_km": 10, "month": null, "region": "Đồng bằng sông Cửu Long", "terrain": "đồng bằng", "preferences": "đồng bằng miền Nam khô ráo"}}
- "Tôi muốn đi biển ở miền Bắc" -> {{"avgtemp_c": 25, "maxwind_kph": 15, "avghumidity": 70, "avgvis_km": 10, "month": null, "region": "Bắc Trung Bộ và Duyên hải miền Trung", "terrain": "ven biển", "preferences": "biển miền Bắc"}}
- "Nơi nóng 30 độ, gió mạnh 25km/h, tầm nhìn xa 15km" -> {{"avgtemp_c": 30, "maxwind_kph": 25, "avghumidity": 70, "avgvis_km": 15, "month": null, "region": null, "terrain": null, "preferences": "nóng 30°C, gió mạnh, tầm nhìn xa"}}

Chỉ trả về JSON, không có text khác.
"""

    def extract_travel_preferences(self, user_input: str) -> Dict:
        """
        Sử dụng OpenAI để trích xuất thông tin du lịch từ input của user
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Bạn là một AI chuyên phân tích yêu cầu du lịch."},
                    {"role": "user", "content": self.extraction_prompt.format(user_input=user_input)}
                ],
                temperature=0.1,  # Giảm temperature để có kết quả ổn định hơn
                max_tokens=500
            )

            # Trích xuất JSON từ response
            content = response.choices[0].message.content.strip()

            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                preferences = json.loads(json_str)

                # Nếu OpenAI không trích xuất được các thông số, thử regex fallback
                if preferences.get('month') is None:
                    month = self._extract_month_fallback(user_input)
                    if month:
                        preferences['month'] = month

                # Fallback cho nhiệt độ
                if preferences.get('avgtemp_c') is None or not (15 <= preferences.get('avgtemp_c', 0) <= 35):
                    temp = self._extract_temperature_fallback(user_input)
                    if temp:
                        preferences['avgtemp_c'] = temp

                # Fallback cho gió - ưu tiên fallback cho các pattern cụ thể
                wind_fallback = self._extract_wind_fallback(user_input)
                if wind_fallback and self._has_specific_wind_pattern(user_input):
                    preferences['maxwind_kph'] = wind_fallback
                elif preferences.get('maxwind_kph') is None or not (5 <= preferences.get('maxwind_kph', 0) <= 30):
                    if wind_fallback:
                        preferences['maxwind_kph'] = wind_fallback

                # Fallback cho độ ẩm
                if preferences.get('avghumidity') is None or not (50 <= preferences.get('avghumidity', 0) <= 90):
                    humidity = self._extract_humidity_fallback(user_input)
                    if humidity:
                        preferences['avghumidity'] = humidity

                # Fallback cho tầm nhìn
                if preferences.get('avgvis_km') is None or not (5 <= preferences.get('avgvis_km', 0) <= 15):
                    visibility = self._extract_visibility_fallback(user_input)
                    if visibility:
                        preferences['avgvis_km'] = visibility

                # Fallback cho vùng miền
                if preferences.get('region') is None:
                    region = self._extract_region_fallback(user_input)
                    if region:
                        preferences['region'] = region

                # Fallback cho địa hình
                if preferences.get('terrain') is None:
                    terrain = self._extract_terrain_fallback(user_input)
                    if terrain:
                        preferences['terrain'] = terrain

                # Validate và set default values
                preferences = self._validate_preferences(preferences)
                return preferences
            else:
                return self._get_default_preferences_with_fallback(user_input)

        except Exception as e:
            print(f"Error in extract_travel_preferences: {e}")
            return self._get_default_preferences_with_fallback(user_input)
    
    def _validate_preferences(self, preferences: Dict) -> Dict:
        """Validate và điều chỉnh các giá trị preferences"""
        validated = {}
        
        # Validate avgtemp_c (15-35)
        validated['avgtemp_c'] = max(15, min(35, preferences.get('avgtemp_c', 25)))
        
        # Validate maxwind_kph (5-30)
        validated['maxwind_kph'] = max(5, min(30, preferences.get('maxwind_kph', 15)))
        
        # Validate avghumidity (50-90)
        validated['avghumidity'] = max(50, min(90, preferences.get('avghumidity', 70)))
        
        # Validate avgvis_km (5-15)
        validated['avgvis_km'] = max(5, min(15, preferences.get('avgvis_km', 10)))
        
        # Validate month (1-12 or None)
        month = preferences.get('month')
        if month is not None:
            validated['month'] = max(1, min(12, int(month)))
        else:
            validated['month'] = None

        # Copy region và terrain
        validated['region'] = preferences.get('region')
        validated['terrain'] = preferences.get('terrain')

        validated['preferences'] = preferences.get('preferences', 'du lịch chung')

        return validated
    
    def _extract_month_fallback(self, user_input: str) -> Optional[int]:
        """Fallback method để trích xuất tháng bằng regex"""
        user_input_lower = user_input.lower()

        # Mapping tháng tiếng Việt - thứ tự quan trọng (từ cụ thể đến chung)
        month_patterns = {
            12: [r'tháng\s*12\b', r'tháng\s*mười\s*hai\b'],
            11: [r'tháng\s*11\b', r'tháng\s*mười\s*một\b'],
            10: [r'tháng\s*10\b', r'tháng\s*mười\b(?!\s*(một|hai))'],  # Không match "mười một" hoặc "mười hai"
            1: [r'tháng\s*1\b', r'tháng\s*một\b', r'tháng\s*giêng\b'],
            2: [r'tháng\s*2\b', r'tháng\s*hai\b'],
            3: [r'tháng\s*3\b', r'tháng\s*ba\b'],
            4: [r'tháng\s*4\b', r'tháng\s*tư\b', r'tháng\s*bốn\b'],
            5: [r'tháng\s*5\b', r'tháng\s*năm\b'],
            6: [r'tháng\s*6\b', r'tháng\s*sáu\b'],
            7: [r'tháng\s*7\b', r'tháng\s*bảy\b'],
            8: [r'tháng\s*8\b', r'tháng\s*tám\b'],
            9: [r'tháng\s*9\b', r'tháng\s*chín\b']
        }

        for month, patterns in month_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return month

        return None

    def _has_specific_wind_pattern(self, user_input: str) -> bool:
        """Kiểm tra xem có pattern gió cụ thể không"""
        user_input_lower = user_input.lower()
        specific_patterns = [
            r'không\s*thích\s*gió',
            r'ít\s*gió',
            r'gió\s*nhẹ',
            r'gió\s*mạnh',
            r'\d+\s*km/h',
            r'\d+\s*kmh'
        ]

        for pattern in specific_patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _extract_temperature_fallback(self, user_input: str) -> Optional[float]:
        """Fallback method để trích xuất nhiệt độ bằng regex"""
        user_input_lower = user_input.lower()

        # Tìm số + đơn vị nhiệt độ
        temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:độ|°)\s*c?\b',
            r'(\d+(?:\.\d+)?)\s*degrees?\s*c(?:elsius)?\b',
            r'nhiệt\s*độ\s*(\d+(?:\.\d+)?)',
            r'temperature\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in temp_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                temp = float(match.group(1))
                # Validate range (15-35°C)
                if 15 <= temp <= 35:
                    return temp

        # Tìm từ khóa mô tả
        if re.search(r'mát\s*mẻ|lạnh|se\s*lạnh', user_input_lower):
            return 20.0  # Mát mẻ
        elif re.search(r'nóng|ấm\s*áp', user_input_lower):
            return 30.0  # Nóng
        elif re.search(r'ôn\s*hòa|dễ\s*chịu', user_input_lower):
            return 25.0  # Ôn hòa

        return None

    def _extract_wind_fallback(self, user_input: str) -> Optional[float]:
        """Fallback method để trích xuất tốc độ gió bằng regex"""
        user_input_lower = user_input.lower()

        # Tìm số + đơn vị gió
        wind_patterns = [
            r'(\d+(?:\.\d+)?)\s*km/h\b',
            r'(\d+(?:\.\d+)?)\s*kmh\b',
            r'gió\s*(\d+(?:\.\d+)?)',
            r'wind\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in wind_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                wind = float(match.group(1))
                # Validate range (5-30 km/h)
                if 5 <= wind <= 30:
                    return wind

        # Tìm từ khóa mô tả - thứ tự quan trọng (từ cụ thể đến chung)
        if re.search(r'không\s*thích\s*gió\s*mạnh', user_input_lower):
            return 6.0  # Không thích gió mạnh
        elif re.search(r'không\s*thích\s*gió|ít\s*gió|không\s*gió', user_input_lower):
            return 6.0  # Ít gió
        elif re.search(r'gió\s*nhẹ|gió\s*yếu', user_input_lower):
            return 8.0  # Gió nhẹ
        elif re.search(r'gió\s*mạnh|gió\s*lớn', user_input_lower):
            return 25.0  # Gió mạnh

        return None

    def _extract_humidity_fallback(self, user_input: str) -> Optional[float]:
        """Fallback method để trích xuất độ ẩm bằng regex"""
        user_input_lower = user_input.lower()

        # Tìm số + đơn vị độ ẩm
        humidity_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\b',
            r'(\d+(?:\.\d+)?)\s*phần\s*trăm',
            r'độ\s*ẩm\s*(\d+(?:\.\d+)?)',
            r'humidity\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in humidity_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                humidity = float(match.group(1))
                # Validate range (50-90%)
                if 50 <= humidity <= 90:
                    return humidity

        # Tìm từ khóa mô tả
        if re.search(r'khô\s*ráo|khô|ít\s*ẩm', user_input_lower):
            return 55.0  # Khô ráo
        elif re.search(r'ẩm\s*ướt|ẩm|nhiều\s*ẩm', user_input_lower):
            return 80.0  # Ẩm ướt
        elif re.search(r'vừa\s*phải|bình\s*thường', user_input_lower):
            return 70.0  # Vừa phải

        return None

    def _extract_visibility_fallback(self, user_input: str) -> Optional[float]:
        """Fallback method để trích xuất tầm nhìn bằng regex"""
        user_input_lower = user_input.lower()

        # Tìm số + đơn vị tầm nhìn
        visibility_patterns = [
            r'(\d+(?:\.\d+)?)\s*km\b',
            r'tầm\s*nhìn\s*(\d+(?:\.\d+)?)',
            r'visibility\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in visibility_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                visibility = float(match.group(1))
                # Validate range (5-15 km)
                if 5 <= visibility <= 15:
                    return visibility

        # Tìm từ khóa mô tả
        if re.search(r'tầm\s*nhìn\s*xa|tầm\s*nhìn\s*tốt|trong\s*vắt', user_input_lower):
            return 12.0  # Tầm nhìn tốt
        elif re.search(r'tầm\s*nhìn\s*kém|mù|sương', user_input_lower):
            return 6.0  # Tầm nhìn kém
        elif re.search(r'tầm\s*nhìn\s*bình\s*thường', user_input_lower):
            return 10.0  # Bình thường

        return None

    def _extract_region_fallback(self, user_input: str) -> Optional[str]:
        """Fallback method để trích xuất vùng miền bằng regex"""
        user_input_lower = user_input.lower()

        # Mapping vùng miền tiếng Việt
        region_patterns = {
            "Tây Nguyên": [r'tây\s*nguyên', r'cao\s*nguyên', r'đà\s*lạt', r'buôn\s*ma\s*thuột'],
            "Đồng bằng sông Hồng": [r'đồng\s*bằng\s*sông\s*hồng', r'hà\s*nội', r'hải\s*phòng', r'nam\s*định', r'thái\s*bình'],
            "Đồng bằng sông Cửu Long": [r'đồng\s*bằng\s*sông\s*cửu\s*long', r'mekong', r'cần\s*thơ', r'an\s*giang', r'cà\s*mau', r'bến\s*tre'],
            "Bắc Trung Bộ và Duyên hải miền Trung": [r'miền\s*trung', r'trung\s*bộ', r'phía\s*trung', r'huế', r'đà\s*nẵng', r'hội\s*an', r'nha\s*trang'],
            "Trung du và miền núi Bắc Bộ": [r'miền\s*núi\s*bắc\s*bộ', r'trung\s*du', r'sapa', r'hà\s*giang', r'cao\s*bằng'],
            "Đông Nam Bộ": [r'đông\s*nam\s*bộ', r'tp\s*hồ\s*chí\s*minh', r'sài\s*gòn', r'đồng\s*nai', r'bình\s*dương']
        }

        # Thêm các pattern chung
        if re.search(r'miền\s*bắc|bắc\s*bộ|phía\s*bắc', user_input_lower):
            # Ưu tiên đồng bằng sông Hồng cho miền Bắc
            return "Đồng bằng sông Hồng"
        elif re.search(r'miền\s*nam|nam\s*bộ|phía\s*nam', user_input_lower):
            # Ưu tiên đồng bằng sông Cửu Long cho miền Nam
            return "Đồng bằng sông Cửu Long"

        for region, patterns in region_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return region

        return None

    def _extract_terrain_fallback(self, user_input: str) -> Optional[str]:
        """Fallback method để trích xuất địa hình bằng regex"""
        user_input_lower = user_input.lower()

        # Mapping địa hình tiếng Việt - thứ tự quan trọng
        terrain_patterns = {
            "ven biển": [r'ven\s*biển', r'biển', r'bãi\s*biển', r'tắm\s*biển', r'gần\s*biển', r'du\s*lịch\s*biển'],
            "miền núi": [r'miền\s*núi', r'núi', r'vùng\s*núi', r'cao', r'leo\s*núi', r'trekking', r'hiking'],
            "đồng bằng": [r'đồng\s*bằng', r'bằng\s*phẳng', r'đồng\s*ruộng', r'nông\s*thôn', r'thôn\s*quê']
        }

        for terrain, patterns in terrain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return terrain

        return None

    def _get_default_preferences(self) -> Dict:
        """Trả về preferences mặc định"""
        return {
            'avgtemp_c': 25,
            'maxwind_kph': 15,
            'avghumidity': 70,
            'avgvis_km': 10,
            'month': None,
            'region': None,
            'terrain': None,
            'preferences': 'du lịch chung'
        }

    def _get_default_preferences_with_fallback(self, user_input: str) -> Dict:
        """Trả về preferences mặc định với tất cả fallback extractions"""
        preferences = self._get_default_preferences()

        # Fallback extractions
        month = self._extract_month_fallback(user_input)
        if month:
            preferences['month'] = month

        temp = self._extract_temperature_fallback(user_input)
        if temp:
            preferences['avgtemp_c'] = temp

        wind = self._extract_wind_fallback(user_input)
        if wind:
            preferences['maxwind_kph'] = wind

        humidity = self._extract_humidity_fallback(user_input)
        if humidity:
            preferences['avghumidity'] = humidity

        visibility = self._extract_visibility_fallback(user_input)
        if visibility:
            preferences['avgvis_km'] = visibility

        region = self._extract_region_fallback(user_input)
        if region:
            preferences['region'] = region

        terrain = self._extract_terrain_fallback(user_input)
        if terrain:
            preferences['terrain'] = terrain

        # Cập nhật preferences description
        desc_parts = []
        if month:
            desc_parts.append(f'tháng {month}')
        if temp:
            desc_parts.append(f'{temp}°C')
        if wind:
            desc_parts.append(f'gió {wind}km/h')
        if humidity:
            desc_parts.append(f'độ ẩm {humidity}%')
        if visibility:
            desc_parts.append(f'tầm nhìn {visibility}km')
        if region:
            desc_parts.append(f'{region}')
        if terrain:
            desc_parts.append(f'{terrain}')

        if desc_parts:
            preferences['preferences'] = f'du lịch {", ".join(desc_parts)}'

        return preferences
    
    def generate_response(self, user_input: str, recommendations: List[Dict]) -> str:
        """
        Tạo response tự nhiên dựa trên recommendations
        """
        try:
            # Tạo context từ recommendations
            locations_text = ""
            for i, rec in enumerate(recommendations[:5], 1):
                locations_text += f"{i}. {rec['city']}, {rec['province']} ({rec['region']}) - Tháng {rec['month']}\n"
                locations_text += f"   Nhiệt độ: {rec['avgtemp_c']:.1f}°C, Gió: {rec['maxwind_kph']:.1f}km/h, "
                locations_text += f"Độ ẩm: {rec['avghumidity']:.1f}%, Tầm nhìn: {rec['avgvis_km']:.1f}km\n"
                locations_text += f"   Điểm phù hợp: {rec['score']:.2f}\n\n"
            
            response_prompt = f"""
Dựa trên yêu cầu du lịch: "{user_input}"

Tôi đã tìm được những địa điểm phù hợp sau:

{locations_text}

Hãy viết một phản hồi tự nhiên, thân thiện để giới thiệu những địa điểm này cho người dùng. 
Phản hồi nên:
- Bắt đầu bằng lời chào thân thiện
- Giải thích ngắn gọn tại sao những địa điểm này phù hợp
- Mô tả đặc điểm thời tiết của từng nơi
- Kết thúc bằng lời khuyên hoặc câu hỏi để tiếp tục hỗ trợ
- Nếu người dùng không trả lời không có liên quan đến chủ đề thì hãy từ chối 1 cách lịch sự (Ví dụ: các chủ đề toán học, xã hội,...)

Viết bằng tiếng Việt, tối đa 300 từ.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia tư vấn du lịch thân thiện và am hiểu về Việt Nam."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return self._get_default_response(recommendations)
    
    def _get_default_response(self, recommendations: List[Dict]) -> str:
        """Tạo response mặc định khi OpenAI không khả dụng"""
        if not recommendations:
            return "Xin lỗi, tôi không tìm thấy địa điểm nào phù hợp với yêu cầu của bạn. Bạn có thể thử với điều kiện khác không?"
        
        response = "Dựa trên yêu cầu của bạn, tôi gợi ý những địa điểm sau:\n\n"
        
        for i, rec in enumerate(recommendations[:5], 1):
            response += f"{i}. **{rec['city']}, {rec['province']}** ({rec['region']})\n"
            response += f"   - Thời gian: Tháng {rec['month']}\n"
            response += f"   - Thời tiết: {rec['avgtemp_c']:.1f}°C, gió {rec['maxwind_kph']:.1f}km/h\n"
            response += f"   - Điểm phù hợp: {rec['score']:.2f}/5\n\n"
        
        response += "Bạn có muốn biết thêm thông tin về địa điểm nào không?"
        return response
