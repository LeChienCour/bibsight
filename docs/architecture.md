# Bibsight — Architecture & Business Pipeline

## Flujo completo

### Evento (día de carrera)

```
Drone/cámara → video MP4
       ↓
[LOCAL - RTX 4070]
  YOLO detección personas
  EasyOCR lee bib numbers
  InsightFace extrae embeddings
  Linking bib ↔ embedding ↔ foto_ids
       ↓
  Genera previews con watermark
       ↓
[UPLOAD A AWS]
  s3://race-photos-raw/          ← originales
  s3://race-photos-preview/      ← JPG watermark → CloudFront
  DynamoDB                       ← {bib: [photo_ids, embedding]}
  OpenSearch Serverless          ← índice k-NN de embeddings
```

### Compra (cualquier día post-evento)

```
Corredor entra a web (Next.js)
  └─ busca por bib number → ve sus fotos preview (CloudFront)
  └─ sube selfie → Lambda extrae embedding (InsightFace CPU ~200ms)
                 → busca k-NN en OpenSearch (threshold coseno ~0.4)
                 → muestra fotos coincidentes

Selecciona fotos → Stripe / MercadoPago
       ↓
Lambda confirma pago
       ↓
Emite presigned URL S3 hires (expiración 1h)
       ↓
Corredor descarga originales
```

## Búsqueda por selfie — cómo funciona

1. Usuario sube selfie al frontend
2. API Gateway → Lambda recibe imagen
3. Lambda corre InsightFace `buffalo_l` → embedding 512-d
4. Consulta OpenSearch k-NN contra embeddings indexados del evento
5. Retorna `photo_ids` con similitud > 0.4 (coseno)
6. Frontend muestra previews watermarked de esas fotos

## Revenue estimado (México)

| Concepto | Valor |
|---|---|
| Precio por foto | $50–150 MXN |
| Fotos promedio por corredor | 3–5 |
| Conversión esperada | 30% |
| Carrera típica | 500 corredores |
| Ingreso bruto por evento | ~$30,000 MXN |
| Costos AWS por evento | <$50 USD |
| Frecuencia | 2 eventos/mes |
| **Ingreso mensual estimado** | **~$60,000 MXN (~$3,000 USD)** |

## Gaps a desarrollar

| Prioridad | Componente | Estado |
|---|---|---|
| 1 | Pipeline local (detección + OCR + embedding) | ✅ POC funcional |
| 2 | Generador de previews con watermark | ⬜ Pendiente |
| 3 | Upload automático a S3 post-procesamiento | ⬜ Pendiente |
| 4 | DynamoDB schema + writer | ⬜ Pendiente |
| 5 | OpenSearch índice k-NN + indexer | ⬜ Pendiente |
| 6 | Lambda búsqueda por bib | ⬜ Pendiente |
| 7 | Lambda búsqueda por selfie (embedding) | ⬜ Pendiente |
| 8 | Lambda pago + presigned URL | ⬜ Pendiente |
| 9 | Frontend Next.js | ⬜ Pendiente |
| 10 | Integración Stripe / MercadoPago | ⬜ Pendiente |

## Consideraciones legales (LFPDPPP)

- Embeddings faciales = datos biométricos sensibles
- Consentimiento explícito requerido en inscripción al evento
- Watermark agresivo en previews
- Originales solo via presigned URL con expiración corta post-pago
- No almacenar selfies — solo el embedding temporal en memoria Lambda
