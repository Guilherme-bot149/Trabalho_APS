# pyuic5 designe.ui -o principal.py

from PyQt5 import uic, QtWidgets


def fecharSistema():
    exit()   


app=QtWidgets.QApplication([])
equipe=uic.loadUi("equipe.ui")
equipe.fecharSistema.clicked.connect(fecharSistema)

equipe.show()
app.exec()



