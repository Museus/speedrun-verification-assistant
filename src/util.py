import requests

def prettify_timestamp(seconds):
    time_portions = []

    if seconds > 3600:
        time_portions.append(f"{int(seconds // 3600)}h")
        seconds = seconds % 3600

    if seconds > 60:
        time_portions.append(f"{int(seconds // 60)}m")
        seconds = seconds % 60

    time_portions.append(f"{seconds:.2f}s")

    return " ".join(time_portions)


def get_next_run_from_srcom(skip=0):
    try:
        runs = requests.get("https://speedrun.com/api/v1/runs?game=o1y9okr6&status=new&embed=category,players").json()["data"]
    except Exception:
        print("Failed to get latest runs!")

    if runs:
        return runs[skip]
