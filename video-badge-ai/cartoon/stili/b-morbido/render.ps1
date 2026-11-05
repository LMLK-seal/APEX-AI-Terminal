$edge="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$d="C:\Users\franc\progetto-p-Iva\video-badge-ai\cartoon\stili\b-morbido"
foreach($n in $args){ & $edge --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 --virtual-time-budget=4000 "--screenshot=$d\$n.png" "file:///C:/Users/franc/progetto-p-Iva/video-badge-ai/cartoon/stili/b-morbido/$n.html" 2>&1 | Out-Null }
