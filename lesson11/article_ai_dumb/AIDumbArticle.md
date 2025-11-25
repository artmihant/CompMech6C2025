
# 7 причин, почему ваш ИИ тупит (и как это исправить)

[Use attached catbot character. Scene: programmer at desk with head in hands, monitors showing red errors. Catbot stands next to them holding tangled code like spaghetti, holographic "?" floating above head, blue glow turning orange. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background. 16:9 preview format.]

Работаете с ИИ-ассистентом и чувствуете, что он вас не понимает? Ломает архитектуру, пишет код мимо кассы, а на простые вопросы отвечает какой-то ерундой?

Спокойно. Скорее всего, дело не в нём. Давайте разберёмся, где собака зарыта.



[Use attached catbot character. Scene: catbot scratching head with mitten-hand, looking at crumpled paper with illegible scribbles. Holographic "???" floating above head, confused cat-face expression, blue glow dimming. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

## 1. «Сделай мне красиво»

> Без чёткого ТЗ — результат ХЗ.

Вы-то знаете, что вам нужно. А вот LLM видит только ваши слова. И если в них есть хоть щёлочка для «творческой интерпретации» — будьте уверены, нейросеть туда пролезет. Со всеми вытекающими.

**Как лечить:**

- **Прояснять до старта.** Чувствуете размытость? Попросите ИИ составить план и задать уточняющие вопросы. Заодно сами лучше поймёте задачу.

- **Запретить самодеятельность.** По умолчанию ИИ обучен не надоедать вопросами и сразу кодить. Измените это в правилах проекта (например, в Cursor: `.cursor/rules/`): «Сначала уточни — потом пиши».

- **Помнить про пайплайн.** Требования → Проектирование → Документация → Код → Тесты → Деплой. Не наоборот.

- **Обучить домену.** Вы — носитель уникального знания о том, что именно вам нужно. ИИ знает многое, но нужное вам может быть похоронено где-то в глубинах весов. Дайте краткую справку с определениями и примерами — вытащите знания на поверхность.

## 2. «Вот тебе 200 файлов, разберись»

[Use attached catbot character. Scene: catbot drowning in ocean of floating code files and folders, one mitten-arm reaching up for help. Paper tsunami everywhere. Worried cat-face, ears flattened, holographic "SOS" above head, glow flickering red. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

Ваш проект — это 3 года работы, 10 уровней абстракции и документация уровня «потом допишу». Вы всё знаете наизусть. А ИИ увидел это богатство 5 минут назад.

Представьте джуниора в первый день. Вот примерно так себя чувствует нейросеть, когда вы просите «быстренько добавить фичу».

**Как помочь:**

- **Комментарии.** Код без комментариев — квест для археолога. Попросите ИИ прокомментировать базу, потом поправьте косяки.

- **Карта территории.** Markdown-файлы с описанием архитектуры — это GPS для нейросети. Без карты она заблудится в трёх соснах.

- **Нейминг и рефакторинг.** Если ИИ «плавает» — возможно, стоит причесать код *до* фичи. Говорящие имена творят чудеса.

- **Стандартные паттерны.** MVC, Repository, Factory — всё это ИИ знает и любит. Явно укажите, что используете.

Да, это работа. Но думайте стратегически: день, когда вы создадите карту проекта — это день, когда ИИ начнёт реально помогать.

**А что с конкретной фичей?**

- **Локализуйте.** «Смотри сюда, туда не лезь».
- **Дробите.** «Переписать всё» — это не задача, это приговор.
- **Ревьюйте.** Это теперь ваша работа. Готовьтесь откатывать. Даже рабочий код может быть «грязным».

## 3. «Сделай сложную штуку, которая везде»

[Use attached catbot character. Scene: catbot juggling too many balls, some falling. Balls have icons: gear, {}, database, API. Stressed cat-face, sweat drops, ears twitching, holographic "!!" above head, glow flashing orange. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]


Фича затрагивает 20 файлов. Нейросеть начинает путаться, терять контекст, противоречить сама себе.

Поставьте себя на её место: 10 минут назад она «проснулась», получила роль сеньора с опытом 150 лет, мегабайт документации, 10 мегабайт кода и команду «делай». Немудрено запутаться.

**Рецепт:**

1. Сначала план работ
2. Каждый модуль — отдельно
3. Каждый шаг — проверка тестами и глазами

**Золотое правило:** новая крупная фича = новый чат. Старый контекст замусорен предыдущими попытками. Передавайте в новую сессию только план и актуальный статус.

## 4. «Состояние размазано везде»

[Use attached catbot character. Scene: catbot as detective with tiny deerstalker hat, holding magnifying glass, examining tangled glowing threads between boxes labeled "cache", "DB", "UI", "hooks". Focused cat-face squinting, holographic magnifying glass icon above head. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

Программа = состояние + правила его изменения. Когда состояние раскидано по кэшам, хукам, базе и UI — путаются все. И люди, и машины.

**Если архитектура... своеобразная:**

- Опишите источники истины отдельным документом
- Добавьте схемы Data Flow
- Запретите ИИ неявно менять логику состояния
- Помните OCP: лучше расширять, чем менять
- «Явное лучше неявного» — напоминайте это нейросети

**Особый случай — асинхронность.** Гонки, дедлоки, очереди — здесь ИИ часто лажает. Рисуйте диаграммы, продумывайте защиту от race conditions вместе.

## 5. «Это нельзя трогать, но я не сказал»

[Use attached catbot character. Scene: catbot accidentally stepping on big red "DO NOT TOUCH" button, innocent/guilty cat-face with wide LED eyes, ears perked up in surprise. Warning signs and caution tape around. Holographic "oops" above head, glow turned bright red. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

Вы знаете: эта функция вызывается только после инициализации, этот объект immutable, базу нельзя трогать до старта сервера. ИИ этого не знает. Откуда ему знать?

**Если инварианты нарушаются:**

- Исправляйте и тут же объясняйте почему
- Вынесите все «нельзя» в отдельный документ
- В идеале каждое «нельзя» должно быть контрактом в коде

**Отдельная боль — публичные API.** Обычно ИИ боится их трогать, но иногда увлекается и ломает клиентов. Добавьте правило: «Интерфейсы можно расширять, ломать обратную совместимость — нельзя».

## 6. «Изобрети то, чего не существует»

[Use attached catbot character. Scene: catbot trying to fit square peg into round hole, very determined cat-face with tongue slightly out. Thought bubble shows owl being stretched over a globe. Hopeful expression, ears forward, holographic lightbulb above head, glow pulsing yellow. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

Вы придумали что-то новое? Круто! Но в обучающей выборке этого нет. ИИ очень хочет помочь и что-нибудь выдаст — но это будет галлюцинация или сова, натянутая на глобус.

**Как работать с новым:**

- Декомпозируйте до знакомых «кирпичиков»
- Объясните идею на примерах, опишите интерфейсы
- Попробуйте **Vibe Coding**: пусть набросает черновик MVP. Но будьте готовы выбросить — это только чтобы поймать идею
- Попросите сначала план. Сразу увидите, как ИИ понял вашу гениальную мысль
- Помните: желание угодить + отсутствие знаний = галлюцинации


## 7. Напиши мне криптографию с нуля

[Use attached catbot character. Scene: catbot at blackboard covered in complex math formulas, holding chalk in trembling mitten-hand. Sweating, overwhelmed cat-face with spiral eyes. Wearing crooked Einstein wig falling off one cat ear. Floating equations everywhere, holographic "E=mc³" above head, glow sputtering. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

Хотите, чтобы LLM реализовала сложный численный метод или криптоалгоритм? Ну... удачи. Это как просить художника собрать адронный коллайдер — талант есть, но не тот.

1.  **А оно вам надо?** Серьёзно. Скорее всего, есть готовая библиотека, написанная людьми, которые съели на этом собаку. Золотое правило: не пишите свою криптографию, если вы не криптограф. Даже если очень хочется.
2.  **Нет библиотеки?** Распишите математику в документации: формулы, ссылки на статьи, примеры входа-выхода. Чем подробнее — тем меньше «творчества».
3.  **Тесты — ваша религия.** Чем злее математика, тем больше тестов. Unit, property-based, edge cases — всё в бой.
4.  **Изолируйте зверя.** Отдельный модуль, никаких лишних зависимостей. Пусть ИИ сфокусируется на алгоритме, а не на том, как он вписывается в вашу архитектуру.
5.  **Включите режим думания.** Модели с *reasoning* справляются с математикой заметно лучше.
6.  **Цикл до победы:** Код → Тест → Фейл → Правка доки → Код → ... → Profit!
---

## Вместо заключения

Всё сказанное сводится к последовательности простых шагов:

1. **Контекст.** Дайте карту: документацию, правила, структуру.
2. **План.** Разбейте слона на котлеты.
3. **Спека.** Вход, выход, ограничения — чётко и явно.
4. **Код.** Пусть ИИ пишет.
5. **Ревью.** Вы проверяете. Глазами и тестами.
6. **Итерация.** Не работает? Уточняйте контекст, план и спеку, а не просите «попробуй ещё раз».

**ИИ не заменяет программиста — он заменяет печатание кода.** 

Вы больше не набиваете символы. Вы проектируете, направляете, проверяете. Вы — архитектор и ревьювер. А ИИ — очень быстрый, очень услужливый, иногда очень глупый исполнитель.

Чем лучше вы управляете — тем меньше он тупит.

---

## Бонус: ещё 8 граблей

[Use attached catbot character. Scene: catbot stepping on a garden rake, the handle swinging up to hit its face. Startled cat-face with wide LED eyes, ears flying back, holographic "!" above head, glow flashing yellow. Multiple rakes scattered around on the ground. Style: kawaii-tech, Pixar-level expressiveness, clean vector illustration, humorous exaggerated pose. Clean white background.]

**«Он же сказал, что работает!»**

ИИ может врать. Не со зла — он правда верит в свой ответ. Называется галлюцинация. Особенно опасно с названиями библиотек, параметрами API и «фактами» из интернета. **Правило: доверяй, но проверяй.** Всегда. Запускайте код прежде чем его заливать в репозиторий.

**«Почему ты это сделал?!»**

ИИ редко объясняет свои решения, если не попросить. А когда объясняет — может нафантазировать постфактум. Хотите понять логику? Просите объяснить *до* написания кода. «Как ты планируешь это реализовать?» — волшебный вопрос.

**«Одна модель на все случаи»**

GPT-4o/Grok Code хорош для быстрых задач. Claude — для длинного контекста и рефакторинга. GPT5 — для сложной логики и математики. Gemini 3 Pro — для работы с большими кодовыми базами. Подбирайте инструмент под задачу. Плотник не забивает гвозди отвёрткой.

**«Просто попробуй ещё раз»**

Если ИИ ошибся и вы просто сказали «неправильно, переделай» — он сделает то же самое, но по-другому. Или хуже. Объясните, *что* не так и *почему*. Дайте ошибку из консоли. Покажите ожидаемый результат. Чем точнее фидбек — тем точнее исправление.

**«Документация? У меня есть ИИ!»**

ИИ обучен на данных до какой-то даты. Новые версии библиотек, свежие API — он их не знает. Но притворится, что знает. Используйте MCP (Model Context Protocol) для подключения к актуальной документации. Или просто кидайте ссылки на доки в контекст.

**«На, держи мой .env»**

ИИ — это облачный сервис. Всё, что вы ему отправите, уходит на чужие сервера. API-ключи, пароли, токены — никогда не кидайте их в контекст. Один раз расслабились — и ваш AWS-аккаунт майнит крипту для кого-то другого.

**«Тот же промпт, что и полгода назад»**

Работаете над проектом долго? Сохраняйте удачные промпты. Записывайте, что сработало. Версионируйте системные инструкции как код. Через месяц вы не вспомните, какая магическая формулировка заставила ИИ делать то, что нужно.

**«Контекст бесконечный, лей больше»**

Каждый токен — это деньги и время. Огромный контекст = медленные ответы и пустой кошелёк. Давайте только релевантное. Один точный файл лучше, чем «вот тебе весь проект, разберись».

---

*P.S. Если после всего этого ИИ всё ещё тупит — возможно, пора отдохнуть. Обоим.*


[Humorous illustrated robot mascot named catbot.

DESIGN:
- A small, friendly robot with rounded white and light gray body panels, an expressive LED face on a dark screen, glowing blue eyes, flexible segmented arms, and a slightly clumsy but charming demeanor.

- Cat ears and a feline expression on the faceplate.
- A faceplate with a cat-like muzzle and large, expressive oval LED eyes that can display emotions (joy, confusion, anxiety, thoughtfulness).
- Short arms with three-fingered, mitten-like hands.
- Primary color: matte white with soft neon backlighting that changes color depending on the robot's mood.
- A holographic symbol displays above the robot's head, reflecting the robot's emotions.

STYLE:
- Kawaii style combined with a tech aesthetic
- Pixar-level appeal and expressiveness
- Clean, simple forms without unnecessary detail
- Friendly and approachable, slightly awkward appearance
Keep the proportions: head is about 40% of the height, compact torso, massive feet, antenna with a tiny blinking light.
Humorous tone: light, expressive, exaggerated poses.

IMAGE VIEW:
- Front view, side view, back view, 3/4 view
- 3-4 examples of facial expressions (joy, confusion, disappointment, moment of insight)
- Clean white background
- Uniform lighting from the top left
- No background elements, full focus on the character
]
