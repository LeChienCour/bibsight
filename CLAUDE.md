# CLAUDE.md

Contexto del proyecto para asistentes de IA (Claude Code, Cursor, etc.).

## Resumen

Plataforma para fotografía deportiva (carreras pedestres, ciclismo, triatlón) que automatiza la **detección de corredores, lectura de números de dorsal (bib numbers) y vinculación con rostros** a partir de video/fotos capturados con dron o cámara. El objetivo final es que cada corredor pueda **buscar y comprar sus fotos** ingresando su número de dorsal o subiendo una selfie.

## Objetivo de negocio

- Procesar video/fotos de un evento deportivo.
- Indexar cada foto por `bib_number` y por `face_embedding`.
- Publicar previews con watermark; vender originales en alta resolución.
- Operación: partnership con organizadores de carreras (consentimiento incluido en la inscripción).

## Stack técnico

### Procesamiento (local, Windows + RTX 4070, 32 GB RAM)

- **Lenguaje:** Python 3.11+
- **Detección de personas y dorsales:** Ultralytics YOLO (v8/v11), fine-tuneado con datasets tipo RBNR.
- **Tracking multi-frame:** ByteTrack o BoT-SORT (integrados en Ultralytics).
- **OCR de dorsales:** PaddleOCR (primario), EasyOCR o TrOCR como fallback.
- **Reconocimiento facial:** InsightFace (modelo `buffalo_l`, ArcFace, embeddings 512-d).
- **Clustering facial:** HDBSCAN / DBSCAN con distancia coseno.
- **Cómputo:** CUDA en RTX 4070. Procesamiento batch, sin restricción de tiempo real.

### Cloud (AWS)

- **Almacenamiento:**
  - `s3://race-photos-raw/` — originales (Intelligent-Tiering).
  - `s3://race-photos-preview/` — JPG con watermark, servido vía CloudFront.
  - `s3://race-photos-hires/` — alta resolución, acceso solo por presigned URL post-pago.
- **Metadata:** DynamoDB (`bib_number` PK, lista de `photo_ids`, `face_embedding` opcional).
- **Búsqueda por rostro (selfie):** OpenSearch Serverless con índice k-NN (cuando el catálogo crezca). Inicialmente búsqueda lineal en Lambda.
- **API:** API Gateway + Lambda (Python).
- **Frontend:** Next.js en Amplify o S3+CloudFront.
- **Pagos:** Stripe o MercadoPago; Lambda emite presigned URL al confirmar pago.
- **Auth:** Cognito (opcional para corredores) / acceso público con búsqueda por bib.

## Pipeline de procesamiento

1. **Ingesta** — cargar video MP4 o batch de fotos desde dron/cámara.
2. **Detección por frame** — YOLO detecta `person`, `bib`, `face`.
3. **Asociación espacial** — vincular `bib` y `face` al `person` que los contiene.
4. **Tracking** — asignar `track_id` consistente a cada corredor a lo largo del video.
5. **OCR + voto mayoritario** — leer el dorsal en N frames del mismo `track_id`, quedarse con la moda.
6. **Embedding facial** — promediar embeddings de la cara del track.
7. **Vinculación bib ↔ cara** — construir tabla `{bib_number: (face_embedding, [photo_ids])}`.
8. **Resolución cruzada** — para fotos donde el dorsal no es legible pero sí la cara, asignar `bib_number` por similitud de embedding.
9. **Upload a S3** — subir raw + generar preview con watermark.
10. **Indexado** — escribir metadata en DynamoDB / OpenSearch.

## Estructura esperada del repo

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   ├── ingest/          # carga de video/fotos
│   ├── detection/       # YOLO wrappers, tracking
│   ├── ocr/             # PaddleOCR + post-procesamiento de bib
│   ├── faces/           # InsightFace + clustering
│   ├── linking/         # lógica bib ↔ face
│   ├── storage/         # S3, DynamoDB clients
│   └── pipeline.py      # orquestador
├── models/              # pesos (gitignored)
├── notebooks/           # exploración
├── infra/               # IaC (Terraform o CDK)
└── frontend/            # Next.js (carpeta separada)
```

## Convenciones

- **Python:** PEP 8, type hints obligatorios, `ruff` + `black`.
- **Logging:** `structlog` con JSON output (compatible CloudWatch).
- **Config:** `pydantic-settings`, variables de entorno con prefijo `RPT_`.
- **AWS:** preferir VPC Endpoints, instancias ARM (Graviton) cuando aplique, SSM Parameter Store para secretos no críticos, Secrets Manager para credenciales.
- **IaC:** Terraform preferido (consistencia con el resto de proyectos del autor).
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`).

## Consideraciones legales (México)

- LFPDPPP: los embeddings faciales son **datos biométricos sensibles**.
- Requiere aviso de privacidad y consentimiento explícito del corredor (idealmente en la inscripción del evento).
- Watermark agresivo en previews; los originales solo se entregan post-pago vía presigned URL con expiración corta.

## Estado actual

- Fase: diseño / POC inicial.
- Primer hito: script end-to-end que procese un video de prueba y genere un JSON con `{frame, track_id, bib_number, face_embedding}` — sin AWS todavía.

## Preferencias del autor para asistentes de IA

- Respuestas **cortas y directas**, con referencias a fuentes cuando aplique.
- Feedback honesto, incluso si es crítico.
- El autor es DevOps/Cloud Architect en AWS; **no explicar conceptos básicos de AWS**.
- Python como lenguaje principal; NodeJS aceptable para frontend/lambdas ligeras.
- Priorizar soluciones AWS-native antes que third-party cuando haya paridad.