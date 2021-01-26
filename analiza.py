import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys

def create_df():
    df = pd.read_excel(str(pathlib.Path().absolute())+"\\Belgium.xlsx")
    return df

def arima_model(df):
    model = ARIMA(df.total_cases, order=(3, 2, 1))
    model_fit = model.fit(disp=0)
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Reszty", ax=ax[0])
    residuals.plot(kind='kde', title='Gęstość', ax=ax[1])
    plt.show()
    # Actual vs Fitted
    model_fit.plot_predict(dynamic=False)
    plt.show()
    # Create Training and Test
    train = df.total_cases[:270]
    test = df.total_cases[270:]
    # Build Model
    model = ARIMA(train, order=(3, 2, 1))
    # model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit(disp=-1)
    # Forecast
    fc, se, conf = fitted.forecast(43, alpha=0.05)  # 95%
    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(train, label='Dane treningowe')
    plt.plot(test, label='Dane rzeczywiste')
    plt.plot(fc_series, label='Prognoza')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Prognoza vs Rzeczywiste')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


def opis(self, df):
    msg = QMessageBox()
    msg.setWindowTitle("Statystyki opisowe")
    msg.setText("Statystyki opisowe dotyczące ilości codziennych zachorowań")
    msg.setGeometry(400, 400, 2000, 1000)
    msg.setDetailedText("Twoje statystyki opisowe to:\n" + stat(df))
    x = msg.exec_()


def stat(df):
    minimum = np.min(df.total_cases)
    maximum = np.max(df.total_cases)
    q1 = np.quantile(df.total_cases, 0.25)
    q2 = np.quantile(df.total_cases, 0.5)
    q3 = np.quantile(df.total_cases, 0.75)
    return "Min: " + str(minimum) + "\nMax: " + str(maximum) + "\nQ1: " + str(q1) + "\nMediana: " + str(
        q2) + "\nQ3: " + str(q3)


def geometryczny(df):
    a = 1
    r = 1.15
    n = 301
    y = np.linspace(13, n, n, dtype=int)
    i = 1
    m = []
    while i <= n:
        wyraz = a * (r ** (i / 3) - 1) / (r - 1)
        m.append(wyraz)
        i += 1
    sns.lineplot(x=y, y=np.log(df.total_cases.iloc[12:]), label='Rzeczywiste')
    sns.lineplot(x=y, y=np.log(m), label='Ciąg geometryczny q =' + str(r))
    plt.xlabel("Dni")
    plt.ylabel("Całkowita liczba zakażeń")
    plt.title('Wykres logarytmiczny całkowitej ilosci zakażeń')
    plt.legend()
    plt.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analiza sytuacji pandemicznej w Belgii")
        self.setGeometry(200, 200, 1000, 600)
        self.setStyleSheet("background-color: light blue;")
        #
        etykieta3 = QLabel(self)
        etykieta3.setGeometry(0, 0, 1500, 1500)
        etykieta3.setStyleSheet("background-image: url(photo2.jpg);")  # zmiana zdj
        self.interfejs()

    def do_plot(self):
        self.wykres = str(self.comboBox1.currentText())
        df = create_df()
        self.case_plot(df)

    def do_analizy(self):
        self.analiza = str(self.comboBox2.currentText())
        df = create_df()
        self.analizy(df)

    def clicked_plot(self):
        # self.close()
        self.do_plot()

    def finish(self):
        self.close()

    def interfejs(self):
        # przyciski, etykiety, rozmieszczenie
        title = QLabel("Aplikacja pozwalająca na analizę sytuacji pandemii COVID-19 w Belgii", self)
        title.setWordWrap(True)
        title.setGeometry(300, 20, 400, 60)
        title.setFont(QFont('Arial', 17))

        etykieta1 = QLabel("Wybierz wizualizację danych", self)
        etykieta1.setWordWrap(True)
        etykieta1.setGeometry(120, 130, 450, 50)
        etykieta1.setFont(QFont('Arial', 14))
        self.comboBox1 = QComboBox(self)
        self.comboBox1.addItems(['Wykres liniowy ilości przypadków zakażenia',
                                 'Wykres liniowy ilości zgonów spowodowanych zakażeniem',
                                 'Wykres liniowy ilości hospitalizowanych pacjentów'])
        self.comboBox1.setGeometry(100, 200, 300, 30)
        self.comboBox1.setFixedSize(350, 80)
        #        self.comboBox1.setStyleSheet("background-color: red;")
        self.comboBox1.setFont(QFont('Arial Black', 10))
        etykieta2 = QLabel("Wybierz metodę analizy danych", self)
        etykieta2.setWordWrap(True)
        etykieta2.setGeometry(610, 130, 450, 50)
        etykieta2.setFont(QFont('Arial', 14))
        self.comboBox2 = QComboBox(self)
        self.comboBox2.addItems(['Analiza szeregu geometrycznego',
                                 'Model ARIMA',
                                 'Podstawowe statystki opisowe'])
        self.comboBox2.setGeometry(580, 200, 300, 30)
        self.comboBox2.setFixedSize(350, 80)
        self.comboBox2.setFont(QFont('Arial Black', 10))
        #        self.comboBox2.setStyleSheet("background-color: red;")
        okButton1 = QPushButton("Pokaż wykres", self)
        okButton1.setGeometry(50, 350, 400, 30)
        okButton1.clicked.connect(self.clicked_plot)
        okButton1.setFixedSize(420, 40)
        okButton1.setFont(QFont('Arial Black', 10))
        #        okButton1.setStyleSheet("background-color: red;")
        okButton2 = QPushButton("Pokaż wyniki analizy", self)
        okButton2.setGeometry(550, 350, 400, 30)
        okButton2.clicked.connect(self.do_analizy)
        okButton2.setFixedSize(420, 40)
        okButton2.setFont(QFont('Arial Black', 10))
        #        okButton2.setStyleSheet("background-color: red;")
        closeButton = QPushButton("Zamknij", self)
        closeButton.setGeometry(800, 500, 50, 30)
        closeButton.clicked.connect(self.finish)
        closeButton.setFixedSize(120, 60)
        closeButton.setFont(QFont('Arial Black', 10))

    #        closeButton.setStyleSheet("background-color: red;")

    def case_plot(self, df):
        if self.wykres == 'Wykres liniowy ilości przypadków zakażenia':
            data = df['total_cases'].values
        elif self.wykres == 'Wykres liniowy ilości zgonów spowodowanych zakażeniem':
            data = df['total_deaths'].values
        elif self.wykres == 'Wykres liniowy ilości hospitalizowanych pacjentów':
            data = df['hosp_patients'].values
        plt.plot(data)
        plt.ylabel('Ilość osób')
        plt.xlabel('Dzień epidemii')
        if self.wykres == 'Wykres liniowy ilości przypadków zakażenia':
            plt.title('Wykres liniowy ilości przypadków zakażenia')
        elif self.wykres == 'Wykres liniowy ilości zgonów spowodowanych zakażeniem':
            plt.title('Wykres liniowy ilości zgonów spowodowanych zakażeniem')
        elif self.wykres == 'Wykres liniowy ilości hospitalizowanych pacjentów':
            plt.title('Wykres liniowy ilości hospitalizowanych pacjentów')
        plt.show()

    def analizy(self, df):
        if self.analiza == 'Analiza szeregu geometrycznego':
            label1 = QLabel("Tutaj musi się pokazac wartość q i fajnie by bylo jakby wraz z wykresem", self)
            geometryczny(df)
            label1.setWordWrap(True)
            label1.setGeometry(500, 400, 200, 70)
        elif self.analiza == 'Model ARIMA':
            label2 = QLabel("Tu musi sie pokazac model arima", self)
            arima_model(df)
            label2.setWordWrap(True)
            label2.setGeometry(500, 400, 200, 70)
        elif self.analiza == 'Podstawowe statystki opisowe':
            label4 = QLabel("Tu sie pokazuja jakies min, max, rozstep, kwantyle, cos jeszcze? ", self)
            opis(self, df)
            label4.setWordWrap(True)
            label4.setGeometry(500, 400, 200, 70)


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec_()




