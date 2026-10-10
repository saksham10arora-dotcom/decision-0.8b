# Records assets/demo.gif. Run from the repo root: vhs assets/demo.tape
Output assets/demo.gif
Set Shell "bash"
Set FontSize 16
Set Width 1120
Set Height 800
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Hide
Type "source .venv/bin/activate && export PS1='$ ' && clear"
Enter
Sleep 1s
Show
Sleep 500ms
Type "python decide.py --demo"
Sleep 600ms
Enter
Wait+Screen@60s /forward pass, \d+ ms\s*\n\s*\n?.*\$/
Sleep 6s
