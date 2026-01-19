"""
Скрипт для проверки подключения к backend API
"""
import requests
import sys

API_BASE = "http://localhost:8000"

def test_connection():
    print("Проверка подключения к backend...")
    print(f"URL: {API_BASE}\n")
    
    try:
        # Проверка health endpoint
        print("1. Проверка health endpoint...")
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Backend доступен: {response.json()}")
        else:
            print(f"   ✗ Неожиданный статус: {response.status_code}")
            return False
        
        # Проверка history endpoint
        print("\n2. Проверка history endpoint...")
        response = requests.get(f"{API_BASE}/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            items_count = len(data.get("items", []))
            print(f"   ✓ История доступна ({items_count} записей)")
        else:
            print(f"   ✗ Ошибка: {response.status_code}")
            return False
        
        print("\n" + "="*50)
        print("✓ Все проверки пройдены!")
        print("✓ Backend готов к работе")
        print("="*50)
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*50)
        print("✗ ОШИБКА: Не удалось подключиться к backend")
        print("\nУбедитесь, что backend запущен:")
        print("  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
        print("="*50)
        return False
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
