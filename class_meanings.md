# data.yaml class nomlari ma'nosi

`data.yaml` ichida `nc: 168` deb yozilgani model 168 xil yo'l belgisi klassini aniqlashga o'rgatilganini bildiradi.

Class nomlari rasmiy yo'l belgisi kodlari ko'rinishida berilgan:

- `I-*` - ogohlantiruvchi belgilar
- `II-*` - imtiyoz / ustuvorlik belgilari
- `III-*` - taqiqlovchi belgilar
- `IV-*` - buyuruvchi belgilar
- `V-*` - axborot-ko'rsatish belgilari
- `VI-*` - servis / xizmat ko'rsatish belgilari
- `VII-*` - qo'shimcha axborot tablichkalari

Masalan:

- `I-1` - I guruhdagi 1-belgi, ya'ni ogohlantiruvchi belgilar turkumidagi belgi
- `III-10` - III guruhdagi 10-belgi, ya'ni taqiqlovchi belgilar turkumidagi belgi
- `IV-4-2` - IV guruhdagi buyuruvchi belgilar turkumidagi aniq kod
- `VII-6-4` - qo'shimcha axborot tablichkalari turkumidagi aniq kod

Bu nomlar model uchun label sifatida ishlatilgan. Ya'ni model rasmda belgi topsa, masalan `III-10` deb qaytaradi; uning to'liq nomini bilish uchun shu kodni yo'l harakati qoidalaridagi belgilar jadvalidan qarash kerak.
