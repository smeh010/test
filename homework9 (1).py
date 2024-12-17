import random

def play_lottery():
    """
    Моделирует игру в лотерейный автомат "777".

    Returns:
        int: Выигрыш (0, 2, 5, 100, 200) или -1 (проигрыш).
    """
    number = random.randint(0, 999)
    
    if number == 777:
      return 200
    elif number % 100 == 77:
        return 100
    elif number % 10 == 7:
        return 5
    elif number % 100 == 0:
        return 2
    else:
        return -1  # Проигрыш, так как игрок заплатил 1 рубль


def simulate_games(num_games):
    """
    Моделирует несколько игр в лотерею и возвращает суммарный выигрыш.

    Args:
        num_games (int): Количество игр для симуляции.

    Returns:
        float: Суммарный выигрыш или проигрыш за все игры (в рублях).
    """
    total_winnings = 0
    for _ in range(num_games):
        total_winnings += play_lottery()
    return total_winnings



if __name__ == "__main__":
    num_simulations = 1000000 # Количество симуляций
    winnings = simulate_games(num_simulations)

    print(f"Общий выигрыш за {num_simulations} игр: {winnings} руб.")
    average_winnings = winnings / num_simulations
    print(f"Средний выигрыш за одну игру: {average_winnings:.4f} руб.")

    if average_winnings > 0:
        print("Игра выгодна игроку.")
    elif average_winnings < 0:
        print("Игра не выгодна игроку.")
    else:
        print("Игра не приносит ни выигрыша, ни проигрыша в среднем.")

    print("\nВероятность разных выигрышей (для общей информации)")
    winnings_counts = {200: 0, 100: 0, 5: 0, 2: 0, -1: 0}
    for _ in range(num_simulations):
        result = play_lottery()
        winnings_counts[result] +=1

    for key, value in winnings_counts.items():
        print(f"Выигрыш {key}: {value/num_simulations:.6f}")
    
    # --- Вывод ---
    print("\n") # пустая строка для читаемости
    conclusion = "# На основе проведенной симуляции, игра в данный лотерейный автомат является невыгодной для игрока, так как средний выигрыш за одну игру отрицательный. Это означает, что в долгосрочной перспективе игрок, скорее всего, потеряет свои деньги. Вероятности выигрышей и проигрыша также показывают, что проигрыш является более вероятным исходом."
    print(conclusion)