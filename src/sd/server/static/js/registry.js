// Реестр экранов: единственное место, где перечислены разделы. Новый раздел — одна запись здесь и файл в screens/.
// Поля: id, path (с параметрами «:id?»), title, icon, role (минимальная), group (work|dev|system), nav (показывать в меню), load (ленивый импорт), badge ('new' — число ждущих тревог).
export const GROUPS = { work: "Работа", dev: "Разработка", system: "Система" };

export const SCREENS = [
  { id: "overview", path: "/", title: "Обзор", icon: "grid", role: "viewer", group: "work", nav: true, load: () => import("./screens/overview.js") },
  { id: "alerts", path: "/alerts", title: "Тревоги", icon: "bell", role: "viewer", group: "work", nav: true, badge: "new", load: () => import("./screens/alerts.js") },
  { id: "review", path: "/review/:id?", title: "Проверка", icon: "eye", role: "viewer", group: "work", nav: true, navPath: "/review", load: () => import("./screens/review.js") },
  { id: "map", path: "/map", title: "Карта", icon: "pin", role: "viewer", group: "work", nav: true, load: () => import("./screens/map.js") },
  { id: "reports", path: "/reports", title: "Отчёты", icon: "file", role: "viewer", group: "work", nav: true, load: () => import("./screens/reports.js") },
  { id: "analyze", path: "/analyze", title: "Разбор видео", icon: "video", role: "viewer", group: "work", nav: true, load: () => import("./screens/analyze.js") },
  { id: "experiments", path: "/experiments", title: "Эксперименты", icon: "chart", role: "developer", group: "dev", nav: true, load: () => import("./screens/experiments.js") },
  { id: "designer", path: "/designer/:name?", title: "Конструктор", icon: "layers", role: "developer", group: "dev", nav: true, navPath: "/designer", load: () => import("./screens/designer.js") },
  { id: "labeling", path: "/labeling", title: "Разметка", icon: "tag", role: "developer", group: "dev", nav: true, load: () => import("./screens/labeling.js") },
  { id: "admin", path: "/admin", title: "Администрирование", icon: "shield", role: "admin", group: "system", nav: true, load: () => import("./screens/admin.js") },
  { id: "account", path: "/account", title: "Мой профиль", icon: "settings", role: "viewer", group: "system", nav: false, load: () => import("./screens/account.js") },
];

/** Нижняя панель на телефоне: самые частые разделы, остальное — через «Ещё». */
export const BOTTOM = ["overview", "alerts", "review", "map"];
