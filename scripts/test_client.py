#!/usr/bin/env python3
"""
Cliente de prueba para Text Evaluation Service
Prueba completa del flujo: submit -> stream -> results
"""

import requests
import json
import time
import sys
import argparse
from typing import Dict, Any, List
from urllib.parse import urlparse


class Colors:
    """Colores para terminal"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.END} {msg}")


def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")


def log_result(msg: str):
    print(f"{Colors.BLUE}[RESULT]{Colors.END} {msg}")


class TextEvaluationClient:
    """Cliente para probar el servicio de evaluación"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Verificar estado del servicio"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log_error(f"Health check falló: {e}")
            return {}
    
    def submit_evaluation(self, items: List[Dict[str, str]]) -> Dict[str, Any]:
        """Enviar textos para evaluación"""
        payload = {"items": items}
        
        try:
            response = self.session.post(
                f"{self.base_url}/evaluate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log_error(f"Evaluación falló: {e}")
            return {}
    
    def stream_results(self, job_id: str, timeout: int = 300):
        """Stream resultados via SSE"""
        stream_url = f"{self.base_url}/stream/{job_id}"
        
        try:
            with self.session.get(stream_url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: '
                            yield data
                        except json.JSONDecodeError:
                            log_warn(f"No se pudo parsear: {line}")
                            
        except requests.RequestException as e:
            log_error(f"Stream falló: {e}")


def get_test_data() -> List[Dict[str, str]]:
    """Datos de prueba realistas"""
    return [
        {
            "id_alumno": "TEST_001",
            "curso": "3r ESO",
            "consigna": "Explica què és la fotosíntesi i per què és important per a la vida.",
            "respuesta": "La fotosíntesi és el procés pel qual les plantes verdes fabriquen el seu propi aliment utilitzant la llum solar, l'aigua i el diòxid de carboni de l'aire. Aquest procés és molt important perquè produeix oxigen que necessitem per respirar i també serveix com a base de la cadena alimentària."
        },
        {
            "id_alumno": "TEST_002",
            "curso": "3r ESO",
            "consigna": "Descriu el cicle de l'aigua i els seus principals components.",
            "respuesta": "L'aigua s'evapora dels oceans i rius per la calor del sol, després forma núvols al cel i quan plou torna a la terra."
        },
        {
            "id_alumno": "TEST_003",
            "curso": "4t ESO",
            "consigna": "Explica les causes i conseqüències del canvi climàtic.",
            "respuesta": "El canvi climàtic passa perquè hi ha massa CO2 a l'atmosfera per culpa de les fàbriques i els cotxes. Això fa que faci més calor i que el gel dels pols es fongui, augmentant el nivell del mar. També hi ha més sequeres i inundacions."
        },
        {
            "id_alumno": "TEST_004",
            "curso": "2n ESO",
            "consigna": "Què són els animals vertebrats? Dona exemples.",
            "respuesta": "Els animals vertebrats són els que tenen esquelet intern amb columna vertebral. Per exemple els peixos, les aus, els mamífers com els gossos i gats, els rèptils com les serps i els amfibis com les granotes."
        },
        {
            "id_alumno": "TEST_005",
            "curso": "1r ESO",
            "consigna": "Explica què són les cèl·lules.",
            "respuesta": "Les cèl·lules són molt petites i no es veuen. Tots els éssers vius en tenen."
        }
    ]


def run_basic_test(client: TextEvaluationClient):
    """Ejecutar test básico con pocos elementos"""
    log_info("🧪 Ejecutando test básico (2 elementos)")
    
    # Usar solo 2 elementos para test rápido
    test_data = get_test_data()[:2]
    
    # 1. Health check
    log_info("1. Verificando health...")
    health = client.health_check()
    if not health:
        log_error("Health check falló")
        return False
        
    log_result(f"Status: {health.get('status', 'unknown')}")
    log_result(f"GPU disponible: {health.get('gpu_available', False)}")
    log_result(f"Modelo cargado: {health.get('model_loaded', False)}")
    
    if not health.get('model_loaded', False):
        log_error("Modelo no está cargado")
        return False
    
    # 2. Submit evaluation
    log_info("2. Enviando evaluación...")
    job_data = client.submit_evaluation(test_data)
    if not job_data:
        return False
        
    job_id = job_data.get('job_id')
    log_result(f"Job ID: {job_id}")
    log_result(f"Total items: {job_data.get('total_items', 0)}")
    log_result(f"Tiempo estimado: {job_data.get('estimated_time_seconds', 0)}s")
    
    # 3. Stream results
    log_info("3. Recibiendo resultados...")
    results_received = 0
    
    for event in client.stream_results(job_id):
        event_type = event.get('event', 'unknown')
        
        if event_type == 'batch_complete':
            batch_data = event.get('data', {})
            batch_results = batch_data.get('results', [])
            progress = batch_data.get('progress', {})
            
            results_received += len(batch_results)
            
            log_result(f"Batch {batch_data.get('batch_number', '?')} completado:")
            for result in batch_results:
                log_result(f"  - {result.get('id_alumno', '?')}: Nota {result.get('nota', 0)}")
                
            log_result(f"Progreso: {progress.get('percentage', 0):.1f}% ({progress.get('completed', 0)}/{progress.get('total', 0)})")
            
        elif event_type == 'complete':
            final_data = event.get('data', {})
            log_result(f"✅ Evaluación completada!")
            log_result(f"Total procesado: {final_data.get('total_processed', 0)}")
            log_result(f"Tiempo total: {final_data.get('total_time_seconds', 0):.1f}s")
            break
            
        elif 'error' in event:
            log_error(f"Error en stream: {event.get('error', 'unknown')}")
            return False
    
    log_info(f"✅ Test básico completado. Resultados recibidos: {results_received}")
    return True


def run_full_test(client: TextEvaluationClient):
    """Ejecutar test completo con todos los elementos"""
    log_info("🧪 Ejecutando test completo (5 elementos)")
    
    test_data = get_test_data()
    
    # Submit evaluation
    log_info("Enviando evaluación completa...")
    job_data = client.submit_evaluation(test_data)
    if not job_data:
        return False
        
    job_id = job_data.get('job_id')
    log_result(f"Job ID: {job_id}")
    
    # Stream results con timeout mayor
    log_info("Recibiendo resultados...")
    all_results = []
    
    for event in client.stream_results(job_id, timeout=600):  # 10 min timeout
        event_type = event.get('event', 'unknown')
        
        if event_type == 'batch_complete':
            batch_data = event.get('data', {})
            batch_results = batch_data.get('results', [])
            all_results.extend(batch_results)
            
            progress = batch_data.get('progress', {})
            log_result(f"Progreso: {progress.get('percentage', 0):.1f}% - ETA: {progress.get('estimated_remaining_seconds', 0):.0f}s")
            
        elif event_type == 'complete':
            log_result("✅ Test completo finalizado!")
            break
    
    # Mostrar resumen final
    log_info("📊 Resumen de resultados:")
    for result in all_results:
        log_result(f"{result.get('id_alumno', '?')}: Nota {result.get('nota', 0)} - {result.get('feedback', '')[:60]}...")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Cliente de prueba para Text Evaluation Service')
    parser.add_argument('--url', default='http://localhost:8000', help='URL base del servicio')
    parser.add_argument('--test', choices=['basic', 'full', 'health'], default='basic', help='Tipo de test')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout en segundos')
    
    args = parser.parse_args()
    
    # Validar URL
    try:
        parsed_url = urlparse(args.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("URL inválida")
    except Exception as e:
        log_error(f"URL inválida: {args.url} - {e}")
        sys.exit(1)
    
    client = TextEvaluationClient(args.url)
    
    log_info(f"🚀 Iniciando pruebas contra {args.url}")
    
    if args.test == 'health':
        # Solo health check
        health = client.health_check()
        if health:
            print(json.dumps(health, indent=2))
            sys.exit(0 if health.get('status') == 'healthy' else 1)
        else:
            sys.exit(1)
            
    elif args.test == 'basic':
        success = run_basic_test(client)
        sys.exit(0 if success else 1)
        
    elif args.test == 'full':
        success = run_full_test(client)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_warn("\n🛑 Test interrumpido por usuario")
        sys.exit(130)
    except Exception as e:
        log_error(f"Error inesperado: {e}")
        sys.exit(1)