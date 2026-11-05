@echo off
rem Render sheet.png e depth.png con Edge headless (stessa cartella)
set EDGE="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
%EDGE% --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 --virtual-time-budget=4000 --screenshot="%~dp0sheet.png" "file:///%~dp0sheet.html"
%EDGE% --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 --virtual-time-budget=4000 --screenshot="%~dp0depth.png" "file:///%~dp0depth.html"
