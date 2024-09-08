
def format_scientific(num):
    return "{:.3e}".format(num).replace("e", " x 10^")

def format_percentage(num):
    return "{:.2f}%".format(num * 100)

def format_seconds_to_hms_string(seconds):
    m, s = divmod(seconds, 60) # divmod(a, b) returns (a // b, a % b)
    h, m = divmod(m, 60)
    # Ignore hours and minutes if they are zero
    if h == 0:
        if m == 0:
            return f"{s:.2f}s"
        return f"{m:.0f}m {s:.2f}s"
    return f"{h:.0f}h {m:.0f}m {s:.2f}s"