import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz;

class HomeController extends ChangeNotifier {
  DateTime? sobrietyStartDate;
  int currentMilestone = 1;
  bool hasCheckedInToday = false;
  int totalCheckIns = 0;
  DateTime? lastCheckInDate;
  late SharedPreferences prefs;
  final List<int> milestones = [1, 7, 14, 21, 30, 60, 90, 180, 365];
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  String? userId;
  Timer? _dateChecker;
  int? _lastCelebratedDays;

  // Notification related properties
  late FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin;
  bool notificationsEnabled = true;
  List<TimeOfDay> dailyReminderTimes = [];

  Future<void> initialize() async {
    prefs = await SharedPreferences.getInstance();
    await _initializeNotifications();
    await _getUserId();

    if (userId != null) {
      await _syncDataFromFirebase();
      String? startDateStr = prefs.getString('sobriety_start_date');
      String? lastCheckInStr = prefs.getString('last_check_in_date');
      sobrietyStartDate =
          startDateStr != null ? DateTime.parse(startDateStr) : null;
      lastCheckInDate =
          lastCheckInStr != null ? DateTime.parse(lastCheckInStr) : null;
      totalCheckIns = prefs.getInt('total_check_ins') ?? 0;
      hasCheckedInToday = _hasCheckedInToday();
      _lastCelebratedDays = prefs.getInt('last_celebrated_days');

      // Load notification settings
      notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
      dailyReminderTimes = _loadReminderTimes();

      _updateMilestone();
      _startDateChecker();
      await _scheduleAllNotifications();
    }
    notifyListeners();
  }

  Future<void> _initializeNotifications() async {
    flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();
    tz.initializeTimeZones();

    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    final DarwinInitializationSettings initializationSettingsDarwin =
        DarwinInitializationSettings();

    final InitializationSettings initializationSettings =
        InitializationSettings(
      android: initializationSettingsAndroid,
      iOS: initializationSettingsDarwin,
    );

    await flutterLocalNotificationsPlugin.initialize(initializationSettings);
  }

  Future<void> _scheduleAllNotifications() async {
    if (!notificationsEnabled) return;

    await flutterLocalNotificationsPlugin.cancelAll();

    for (var reminderTime in dailyReminderTimes) {
      await _scheduleDailyReminder(reminderTime);
    }

    await _scheduleMilestoneNotifications();
    await _scheduleRandomMotivationalNotifications();

    if (!hasCheckedInToday) {
      await _scheduleCheckInReminder();
    }
  }

  Future<void> _scheduleDailyReminder(TimeOfDay time) async {
    final now = DateTime.now();
    var scheduledTime =
        DateTime(now.year, now.month, now.day, time.hour, time.minute);

    if (scheduledTime.isBefore(now)) {
      scheduledTime = scheduledTime.add(const Duration(days: 1));
    }

    await flutterLocalNotificationsPlugin.zonedSchedule(
      1000 + dailyReminderTimes.indexOf(time),
      'Daily Sobriety Reminder',
      'Remember to stay strong and committed to your sobriety journey today!',
      tz.TZDateTime.from(scheduledTime, tz.local),
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'daily_reminder',
          'Daily Sobriety Reminders',
          channelDescription: 'Reminders to stay committed to your sobriety',
          importance: Importance.high,
          priority: Priority.high,
        ),
      ),
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  Future<void> _scheduleMilestoneNotifications() async {
    final upcomingMilestones =
        milestones.where((m) => m > totalCheckIns).toList();

    for (var milestone in upcomingMilestones) {
      final daysToMilestone = milestone - totalCheckIns;
      final milestoneDate = DateTime.now().add(Duration(days: daysToMilestone));

      await flutterLocalNotificationsPlugin.zonedSchedule(
        2000 + milestone,
        'Milestone Approaching!',
        'You\'re ${daysToMilestone} days away from ${milestone} days sober! Keep going!',
        tz.TZDateTime.from(
            milestoneDate.subtract(const Duration(days: 1)), tz.local),
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'milestone_reminder',
            'Milestone Reminders',
            channelDescription:
                'Notifications for upcoming sobriety milestones',
            importance: Importance.high,
          ),
        ),
        androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
      );
    }
  }

  Future<void> _scheduleCheckInReminder() async {
    final now = DateTime.now();
    var eveningTime = DateTime(now.year, now.month, now.day, 20, 0);

    if (eveningTime.isBefore(now)) {
      eveningTime = eveningTime.add(const Duration(days: 1));
    }

    await flutterLocalNotificationsPlugin.zonedSchedule(
      3000,
      'Daily Check-in Reminder',
      'Don\'t forget to check in today and track your sobriety progress!',
      tz.TZDateTime.from(eveningTime, tz.local),
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'checkin_reminder',
          'Check-in Reminders',
          channelDescription: 'Reminders to complete your daily check-in',
          importance: Importance.high,
        ),
      ),
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
    );
  }

  Future<void> _scheduleRandomMotivationalNotifications() async {
    final messages = [
      'You\'re doing amazing! One day at a time.',
      'Remember why you started this journey. You\'ve got this!',
      'Every sober day is a victory. Celebrate your strength!',
    ];

    final random = Random();

    for (int i = 0; i < 2; i++) {
      final daysToAdd = random.nextInt(7);
      final hours = 9 + random.nextInt(10);
      final minutes = random.nextInt(60);

      var notificationTime = DateTime(
        DateTime.now().year,
        DateTime.now().month,
        DateTime.now().day + daysToAdd,
        hours,
        minutes,
      );

      if (notificationTime.isBefore(DateTime.now())) {
        notificationTime = notificationTime.add(const Duration(days: 7));
      }

      await flutterLocalNotificationsPlugin.zonedSchedule(
        4000 + i,
        'Motivational Boost',
        messages[random.nextInt(messages.length)],
        tz.TZDateTime.from(notificationTime, tz.local),
        const NotificationDetails(
          android: AndroidNotificationDetails(
            'motivational',
            'Motivational Messages',
            channelDescription: 'Encouraging messages to support your sobriety',
          ),
        ),
        androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
      );
    }
  }

  Future<void> showMilestoneAchievedNotification(int days) async {
    await flutterLocalNotificationsPlugin.show(
      5000,
      '🎉 Congratulations! 🎉',
      'You\'ve reached $days days of sobriety! Amazing work!',
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'milestone_achieved',
          'Milestone Achieved',
          channelDescription: 'Celebrations for reaching sobriety milestones',
          importance: Importance.max,
          priority: Priority.max,
        ),
      ),
    );
  }

  Future<void> addDailyReminder(TimeOfDay time) async {
    dailyReminderTimes.add(time);
    await _saveReminderTimes();
    await _scheduleDailyReminder(time);
    notifyListeners();
  }

  Future<void> removeDailyReminder(TimeOfDay time) async {
    dailyReminderTimes
        .removeWhere((t) => t.hour == time.hour && t.minute == time.minute);
    await _saveReminderTimes();
    await flutterLocalNotificationsPlugin
        .cancel(1000 + dailyReminderTimes.indexOf(time));
    notifyListeners();
  }

  Future<void> _saveReminderTimes() async {
    final timesAsString =
        dailyReminderTimes.map((t) => '${t.hour}:${t.minute}').toList();
    await prefs.setStringList('daily_reminder_times', timesAsString);
  }

  List<TimeOfDay> _loadReminderTimes() {
    final timesAsString = prefs.getStringList('daily_reminder_times') ?? [];
    return timesAsString.map((s) {
      final parts = s.split(':');
      return TimeOfDay(hour: int.parse(parts[0]), minute: int.parse(parts[1]));
    }).toList();
  }

  Future<void> toggleNotifications(bool enabled) async {
    notificationsEnabled = enabled;
    await prefs.setBool('notifications_enabled', enabled);

    if (enabled) {
      await _scheduleAllNotifications();
    } else {
      await flutterLocalNotificationsPlugin.cancelAll();
    }
    notifyListeners();
  }

  bool shouldCelebrate() {
    if (totalCheckIns == 0) return false;
    if (_lastCelebratedDays == totalCheckIns) return false;

    bool shouldCelebrate = milestones.contains(totalCheckIns);
    if (totalCheckIns > 365) {
      shouldCelebrate = shouldCelebrate || (totalCheckIns % 30 == 0);
    }

    if (shouldCelebrate) {
      _lastCelebratedDays = totalCheckIns;
      prefs.setInt('last_celebrated_days', totalCheckIns);
    }

    return shouldCelebrate;
  }

  Future<void> _getUserId() async {
    User? user = _auth.currentUser;
    userId = user?.uid;
  }

  Future<void> _syncDataFromFirebase() async {
    try {
      if (userId == null) return;

      DocumentSnapshot userDoc =
          await _firestore.collection('users').doc(userId).get();

      if (userDoc.exists) {
        Map<String, dynamic> data = userDoc.data() as Map<String, dynamic>;
        String? firebaseStartDate = data['sobriety_start_date'];
        bool firebaseCheckedInToday = data['has_checked_in_today'] ?? false;
        int firebaseTotalCheckIns = data['total_check_ins'] ?? 0;
        String? firebaseLastCheckIn = data['last_check_in_date'];

        await prefs.setString('sobriety_start_date', firebaseStartDate ?? '');
        await prefs.setInt('total_check_ins', firebaseTotalCheckIns);
        await prefs.setString('last_check_in_date', firebaseLastCheckIn ?? '');
        await prefs.setBool(
          'checkin_${DateFormat('yyyy-MM-dd').format(DateTime.now())}',
          firebaseCheckedInToday,
        );
      } else {
        await _firestore.collection('users').doc(userId).set({
          'sobriety_start_date': '',
          'has_checked_in_today': false,
          'total_check_ins': 0,
          'last_check_in_date': '',
        });
      }
    } catch (e) {
      print('Error syncing data from Firebase: $e');
    }
  }

  Future<void> _syncDataToFirebase() async {
    try {
      if (userId == null) return;

      await _firestore.collection('users').doc(userId).update({
        'sobriety_start_date': sobrietyStartDate?.toIso8601String() ?? '',
        'has_checked_in_today': hasCheckedInToday,
        'total_check_ins': totalCheckIns,
        'last_check_in_date': lastCheckInDate?.toIso8601String() ?? '',
      });
    } catch (e) {
      print('Error syncing data to Firebase: $e');
    }
  }

  bool _hasCheckedInToday() {
    String today = DateFormat('yyyy-MM-dd').format(DateTime.now());
    return prefs.getBool('checkin_$today') ?? false;
  }

  void _startDateChecker() {
    _dateChecker = Timer.periodic(const Duration(minutes: 1), (_) {
      if (_hasDateChanged()) {
        _handleNewDay();
      }
    });
  }

  bool _hasDateChanged() {
    final lastCheck = prefs.getString('last_date_check');
    final currentDate = DateFormat('yyyy-MM-dd').format(DateTime.now());
    return lastCheck != currentDate;
  }

  Future<bool> checkMissedDay(BuildContext context) async {
    if (lastCheckInDate == null) return false;

    final today = DateTime.now();
    final difference = today.difference(lastCheckInDate!).inDays;

    if (difference > 1) {
      bool shouldReset = await _showMissedDayDialog(context);
      return shouldReset;
    }
    return false;
  }

  Future<bool> _showMissedDayDialog(BuildContext context) async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) => AlertDialog(
        title: Text('Missed Check-in', style: GoogleFonts.poppins()),
        content: Text(
          'We noticed you missed checking in yesterday. Were you loyal to your plan, or did you relapse? Please be honest with yourself.',
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('I Relapsed',
                style: GoogleFonts.poppins(color: Colors.red)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text('I Stayed Sober',
                style: GoogleFonts.poppins(color: Colors.green)),
          ),
        ],
      ),
    );

    return result ?? true;
  }

  void _handleNewDay() async {
    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());
    await prefs.setString('last_date_check', today);
    hasCheckedInToday = false;
    notifyListeners();
    await _scheduleAllNotifications();
  }

  void _updateMilestone() {
    if (sobrietyStartDate == null) return;
    int daysSober = totalCheckIns;
    for (int milestone in milestones.reversed) {
      if (daysSober >= milestone) {
        currentMilestone = milestone;
        notifyListeners();
        return;
      }
    }
  }

  Future<void> checkIn(BuildContext context) async {
    if (hasCheckedInToday) return;

    if (await checkMissedDay(context)) {
      await _resetData();
      return;
    }

    final today = DateFormat('yyyy-MM-dd').format(DateTime.now());
    if (sobrietyStartDate == null) {
      sobrietyStartDate = DateTime.parse(today);
      await prefs.setString('sobriety_start_date', today);
    }

    totalCheckIns++;
    lastCheckInDate = DateTime.now();

    await prefs.setInt('total_check_ins', totalCheckIns);
    await prefs.setString(
        'last_check_in_date', lastCheckInDate!.toIso8601String());
    await prefs.setBool('checkin_$today', true);

    hasCheckedInToday = true;

    if (shouldCelebrate()) {
      await showMilestoneAchievedNotification(totalCheckIns);
    }

    _updateMilestone();
    await _syncDataToFirebase();
    await _scheduleAllNotifications();
    notifyListeners();
  }

  int getDaysSober() {
    return totalCheckIns;
  }

  Future<void> handleRelapse(BuildContext context) async {
    final result = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('New Beginning', style: GoogleFonts.poppins()),
        content: Text(
          "It's okay to start fresh. Remember, every day is a new opportunity.",
          style: GoogleFonts.poppins(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child:
                Text('Cancel', style: GoogleFonts.poppins(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Reset', style: GoogleFonts.poppins(color: Colors.red)),
          ),
        ],
      ),
    );

    if (result == true) {
      await _resetData();
    }
  }

  Future<void> _resetData() async {
    await prefs.remove('sobriety_start_date');
    await prefs.remove('last_celebrated_days');
    await prefs.remove('total_check_ins');
    await prefs.remove('last_check_in_date');
    await _resetCheckins();

    sobrietyStartDate = null;
    hasCheckedInToday = false;
    _lastCelebratedDays = null;
    totalCheckIns = 0;
    lastCheckInDate = null;

    await _syncDataToFirebase();
    await _scheduleAllNotifications();
    notifyListeners();
  }

  Future<void> _resetCheckins() async {
    final keys = prefs.getKeys();
    for (String key in keys) {
      if (key.startsWith('checkin_')) {
        await prefs.remove(key);
      }
    }
  }

  @override
  void dispose() {
    _dateChecker?.cancel();
    super.dispose();
  }
}
