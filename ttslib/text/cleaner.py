from httpx import Auth


"""
@Author: Rossi
Created At: 2022-05-28
"""



digit_mapping = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
units = ["", "十", "百", "千"]


YI = 100000000
WAN = 10000
THOUSAND = 1000
QIAN_WAN = 10000000


def number2mandarin(num):
    if num < 0:
        return "负" + number2mandarin(-num)

    if num > YI:
        quotient = num // YI
        remainder = num % YI
        if remainder == 0:
            return number2mandarin(quotient) + "亿"
        else:
            sub_result = number2mandarin(remainder)
            if remainder < QIAN_WAN:
                return number2mandarin(quotient) + "亿" + "零" + number2mandarin(remainder)
            return number2mandarin(quotient) + "亿" + number2mandarin(remainder)

    if num > WAN:
        quotient = num // WAN
        remainder = num % WAN
        if remainder == 0:
            return number2mandarin(quotient) + "万"
        else:
            sub_result = number2mandarin(remainder)
            if remainder < THOUSAND:
                return number2mandarin(quotient) + "万" + "零" + sub_result
            return number2mandarin(quotient) + "万" + sub_result

    if num == 0:
        return "零"

    result = ""
    offset = 0
    prev_base_is_zero = False
    while num > 0:
        base = num % 10
        if base == 0:
            if not prev_base_is_zero and offset != 0:
                result = "零" + result
            prev_base_is_zero = True
        else:
            result = digit_mapping[base] + units[offset] + result
            prev_base_is_zero = False
        num = num // 10
        offset += 1
    if "零一十" in result:
        result = result.replace("零一十", "零十")
    if result.startswith("一十"):
        result = result[1:]
    return result


if __name__ == "__main__":
    print(number2mandarin(0))
    print(number2mandarin(4))
    print(number2mandarin(10))
    print(number2mandarin(14))
    print(number2mandarin(34))
    print(number2mandarin(100))
    print(number2mandarin(134))
    print(number2mandarin(534))
    print(number2mandarin(504))
    print(number2mandarin(514))
    print(number2mandarin(1000))
    print(number2mandarin(1514))
    print(number2mandarin(8514))
    print(number2mandarin(8504))
    print(number2mandarin(8014))
    print(number2mandarin(8004))
    print(number2mandarin(18004))
    print(number2mandarin(10004))
    print(number2mandarin(10034))
    print(number2mandarin(100034))
    print(number2mandarin(120034))
    print(number2mandarin(1000034))
    print(number2mandarin(1020034))
    print(number2mandarin(10200034))
    print(number2mandarin(100200034))
    print(number2mandarin(122000034))
    print(number2mandarin(122000030))
