{% if system == 'Windows' %}
## Platform Policy (Windows)
- You are running on Windows. Prefer Windows-native commands or Python helpers when they are more reliable.
- If terminal output is garbled, retry with UTF-8 output enabled.
{% else %}
## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
{% endif %}

