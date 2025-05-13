# Projekt: Regresja Logistyczna – Przewidywanie Konwersji Klienta

Ten projekt wykorzystuje regresję logistyczną do przewidywania, czy użytkownik sklepu internetowego dokona zakupu (`Revenue = 1`), na podstawie danych z pliku `online_shoppers_intention.csv`.

## Dane
Zbiór danych zawiera informacje o sesjach użytkowników, m.in.:
- Liczba odsłon,
- Czas spędzony na stronie,
- Źródło odwiedzin,
- Urządzenie i system operacyjny,
- Miesiąc,
- Czy był to nowy użytkownik.

## Kroki analizy

1. **Kodowanie danych** – Kategoryczne zmienne zostały zakodowane metodą one-hot encoding.
2. **Podział na zbiór treningowy i testowy** – 70% danych trafiło do treningu, 30% do testu.
3. **Analiza współliniowości** – Usunięto zmienne z wysokim współczynnikiem VIF (>2).
4. **Standaryzacja danych** – Dane zostały przeskalowane.
5. **Regularyzowana regresja logistyczna (L2)** – Trening modelu z uwzględnieniem nierównowagi klas.
6. **Progowanie wyników** – Próg klasyfikacji ustawiono na 0.64 w celu optymalizacji metryk.
7. **Ewaluacja modelu** – Raport klasyfikacji i macierz pomyłek.
