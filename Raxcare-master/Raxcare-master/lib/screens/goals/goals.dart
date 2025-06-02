import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:convert';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:shared_preferences/shared_preferences.dart';
// ignore: depend_on_referenced_packages
import 'package:intl/intl.dart';

// Enhanced Preferences Service
class PreferencesService {
  static Future<void> saveData(String key, String value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(key, value);
  }

  static Future<String?> getData(String key) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(key);
  }
}

// Goal Model with Additional Features
class Goal {
  String id;
  String description;
  bool isCompleted;
  DateTime createdAt;
  DateTime? completedAt;
  List<String> reflections;
  String? category;
  int priority; // 1-3 where 1 is highest

  Goal({
    required this.description,
    this.isCompleted = false,
    String? id,
    DateTime? createdAt,
    this.completedAt,
    List<String>? reflections,
    this.category,
    this.priority = 2,
  })  : id = id ?? DateTime.now().millisecondsSinceEpoch.toString(),
        createdAt = createdAt ?? DateTime.now(),
        reflections = reflections ?? [];

  factory Goal.fromJson(Map<String, dynamic> json) {
    return Goal(
      id: json['id'],
      description: json['description'],
      isCompleted: json['isCompleted'],
      createdAt: DateTime.parse(json['createdAt']),
      completedAt: json['completedAt'] != null
          ? DateTime.parse(json['completedAt'])
          : null,
      reflections: List<String>.from(json['reflections'] ?? []),
      category: json['category'],
      priority: json['priority'] ?? 2,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'description': description,
      'isCompleted': isCompleted,
      'createdAt': createdAt.toIso8601String(),
      'completedAt': completedAt?.toIso8601String(),
      'reflections': reflections,
      'category': category,
      'priority': priority,
    };
  }

  Color getPriorityColor() {
    switch (priority) {
      case 1:
        return Colors.red.shade400;
      case 2:
        return Colors.orange.shade400;
      case 3:
        return Colors.blue.shade400;
      default:
        return Colors.grey;
    }
  }
}

// Enhanced Goal Provider
class GoalProvider with ChangeNotifier {
  List<Goal> _goals = [];
  bool _loading = true;
  final GenerativeModel? _aiModel;

  List<Goal> get goals => _goals;
  bool get loading => _loading;
  double get completionRate => _goals.isEmpty
      ? 0.0
      : _goals.where((goal) => goal.isCompleted).length / _goals.length;

  GoalProvider({GenerativeModel? aiModel}) : _aiModel = aiModel {
    loadGoals();
  }

  Future<void> loadGoals() async {
    try {
      _loading = true;
      notifyListeners();

      final savedGoalsString =
          await PreferencesService.getData('goals') ?? '[]';
      final List<dynamic> goalList = json.decode(savedGoalsString);
      _goals = goalList.map((goal) => Goal.fromJson(goal)).toList();

      _goals.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    } catch (e) {
      debugPrint('Error loading goals: $e');
      _goals = [];
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> saveGoals() async {
    try {
      final goalsString =
          json.encode(_goals.map((goal) => goal.toJson()).toList());
      await PreferencesService.saveData('goals', goalsString);
    } catch (e) {
      debugPrint('Error saving goals: $e');
    }
  }

  Future<void> addGoal(String description,
      {String? category, int priority = 2}) async {
    if (description.trim().isEmpty) return;

    final goal = Goal(
      description: description.trim(),
      category: category,
      priority: priority,
    );
    _goals.insert(0, goal);
    await saveGoals();
    notifyListeners();
  }

  Future<void> removeGoal(String id) async {
    _goals.removeWhere((goal) => goal.id == id);
    await saveGoals();
    notifyListeners();
  }

  Future<void> toggleGoalCompletion(String id) async {
    final goalIndex = _goals.indexWhere((goal) => goal.id == id);
    if (goalIndex == -1) return;

    _goals[goalIndex].isCompleted = !_goals[goalIndex].isCompleted;
    _goals[goalIndex].completedAt =
        _goals[goalIndex].isCompleted ? DateTime.now() : null;

    await saveGoals();
    notifyListeners();
  }

  Future<void> addReflection(String goalId, String reflection) async {
    final goalIndex = _goals.indexWhere((goal) => goal.id == goalId);
    if (goalIndex == -1) return;

    _goals[goalIndex].reflections.add(reflection);
    await saveGoals();
    notifyListeners();
  }

  Future<String> getAIGuidance(String goalDescription) async {
    if (_aiModel == null) {
      return "AI guidance is currently unavailable. Please try again later.";
    }

    try {
      final prompt = '''
      As a CBT-focused AI assistant, provide brief, actionable guidance for:
      "$goalDescription"
      
      Include:
      1. Potential cognitive distortions
      2. Behavioral activation steps
      3. Thought challenging techniques
      
      Keep response under 150 words.
      ''';

      final content = [Content.text(prompt)];
      final response = await _aiModel!.generateContent(content);
      return response.text ?? "Unable to generate guidance at this moment.";
    } catch (e) {
      debugPrint('Error getting AI guidance: $e');
      return "An error occurred while getting AI guidance. Please try again later.";
    }
  }
}

// Beautiful Main Screen
class GoalsScreen extends StatelessWidget {
  const GoalsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade50,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 200,
            floating: false,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              title: const Text('Goal Mastery',
                  style: TextStyle(color: Colors.white)),
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.deepPurple.shade600,
                      Colors.indigo.shade600,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Consumer<GoalProvider>(
                  builder: (context, provider, _) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 16.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          CircularProgressIndicator(
                            value: provider.completionRate,
                            backgroundColor: Colors.white.withOpacity(0.3),
                            valueColor: const AlwaysStoppedAnimation<Color>(
                                Colors.white),
                            strokeWidth: 8,
                          ),
                          const SizedBox(height: 16),
                          Text(
                            '${(provider.completionRate * 100).toStringAsFixed(0)}% Complete',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: _buildAddGoalCard(context),
            ),
          ),
          Consumer<GoalProvider>(
            builder: (context, provider, _) {
              if (provider.loading) {
                return const SliverFillRemaining(
                  child: Center(child: CircularProgressIndicator()),
                );
              }
              return provider.goals.isEmpty
                  ? SliverFillRemaining(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.flag,
                                size: 60, color: Colors.grey.shade400),
                            const SizedBox(height: 16),
                            Text(
                              'No goals yet',
                              style: TextStyle(
                                fontSize: 18,
                                color: Colors.grey.shade600,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Add your first goal to get started',
                              style: TextStyle(
                                color: Colors.grey.shade500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  : SliverList(
                      delegate: SliverChildBuilderDelegate(
                        (context, index) =>
                            _GoalCard(goal: provider.goals[index]),
                        childCount: provider.goals.length,
                      ),
                    );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAddGoalCard(BuildContext context) {
    final TextEditingController controller = TextEditingController();
    String? selectedCategory;
    int priority = 2;

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'New Goal',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: InputDecoration(
                hintText: 'What do you want to achieve?',
                filled: true,
                fillColor: Colors.grey.shade100,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    decoration: InputDecoration(
                      labelText: 'Category',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 4,
                      ),
                    ),
                    items: const [
                      DropdownMenuItem(value: 'Health', child: Text('Health')),
                      DropdownMenuItem(value: 'Career', child: Text('Career')),
                      DropdownMenuItem(
                          value: 'Relationships', child: Text('Relationships')),
                      DropdownMenuItem(
                          value: 'Personal', child: Text('Personal')),
                    ],
                    onChanged: (value) => selectedCategory = value,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: DropdownButtonFormField<int>(
                    value: priority,
                    decoration: InputDecoration(
                      labelText: 'Priority',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 4,
                      ),
                    ),
                    items: const [
                      DropdownMenuItem(
                        value: 1,
                        child: Row(
                          children: [
                            Icon(Icons.keyboard_double_arrow_up,
                                size: 16, color: Colors.red),
                            SizedBox(width: 4),
                            Text('High'),
                          ],
                        ),
                      ),
                      DropdownMenuItem(
                        value: 2,
                        child: Row(
                          children: [
                            Icon(Icons.keyboard_arrow_up,
                                size: 16, color: Colors.orange),
                            SizedBox(width: 4),
                            Text('Medium'),
                          ],
                        ),
                      ),
                      DropdownMenuItem(
                        value: 3,
                        child: Row(
                          children: [
                            Icon(Icons.keyboard_arrow_down,
                                size: 16, color: Colors.blue),
                            SizedBox(width: 4),
                            Text('Low'),
                          ],
                        ),
                      ),
                    ],
                    onChanged: (value) => priority = value ?? 2,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
                onPressed: () async {
                  if (controller.text.isNotEmpty) {
                    await Provider.of<GoalProvider>(context, listen: false)
                        .addGoal(
                      controller.text,
                      category: selectedCategory,
                      priority: priority,
                    );
                    controller.clear();
                  }
                },
                child: const Text(
                  'Add Goal',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Stunning Goal Card
class _GoalCard extends StatefulWidget {
  final Goal goal;

  const _GoalCard({required this.goal});

  @override
  State<_GoalCard> createState() => _GoalCardState();
}

class _GoalCardState extends State<_GoalCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<GoalProvider>(context, listen: false);
    final goal = widget.goal;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () => setState(() => _expanded = !_expanded),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    GestureDetector(
                      onTap: () => provider.toggleGoalCompletion(goal.id),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        width: 24,
                        height: 24,
                        decoration: BoxDecoration(
                          color: goal.isCompleted
                              ? Colors.green.shade100
                              : Colors.transparent,
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: goal.isCompleted
                                ? Colors.green
                                : Colors.grey.shade400,
                            width: 2,
                          ),
                        ),
                        child: goal.isCompleted
                            ? const Icon(
                                Icons.check,
                                size: 16,
                                color: Colors.green,
                              )
                            : null,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(
                        goal.description,
                        style: TextStyle(
                          fontSize: 16,
                          decoration: goal.isCompleted
                              ? TextDecoration.lineThrough
                              : null,
                          color: goal.isCompleted
                              ? Colors.grey.shade600
                              : Colors.black87,
                        ),
                      ),
                    ),
                    Icon(
                      _expanded ? Icons.expand_less : Icons.expand_more,
                      color: Colors.grey.shade600,
                    ),
                  ],
                ),
                if (goal.category != null || goal.priority != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Row(
                      children: [
                        if (goal.category != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.blue.shade50,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              goal.category!,
                              style: TextStyle(
                                color: Colors.blue.shade800,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        if (goal.category != null && goal.priority != null)
                          const SizedBox(width: 8),
                        if (goal.priority != null)
                          Icon(
                            Icons.circle,
                            size: 12,
                            color: goal.getPriorityColor(),
                          ),
                      ],
                    ),
                  ),
                if (_expanded) ...[
                  const SizedBox(height: 16),
                  _buildDetailsSection(context, goal),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDetailsSection(BuildContext context, Goal goal) {
    final provider = Provider.of<GoalProvider>(context, listen: false);
    final dateFormat = DateFormat('MMM d, y');

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.calendar_today, size: 16, color: Colors.grey.shade600),
            const SizedBox(width: 8),
            Text(
              'Created: ${dateFormat.format(goal.createdAt)}',
              style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
            ),
            if (goal.completedAt != null) ...[
              const SizedBox(width: 16),
              Icon(Icons.flag, size: 16, color: Colors.grey.shade600),
              const SizedBox(width: 8),
              Text(
                'Completed: ${dateFormat.format(goal.completedAt!)}',
                style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
              ),
            ],
          ],
        ),
        const SizedBox(height: 16),
        if (goal.reflections.isNotEmpty) ...[
          const Text(
            'Reflections:',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          ...goal.reflections.map(
            (reflection) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.circle,
                    size: 8,
                    color: Colors.grey.shade500,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      reflection,
                      style: TextStyle(color: Colors.grey.shade800),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            OutlinedButton.icon(
              icon: const Icon(Icons.psychology, size: 16),
              label: const Text('Guidance'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.deepPurple,
                side: BorderSide(color: Colors.deepPurple.shade200),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
              onPressed: () async {
                final guidance = await provider.getAIGuidance(goal.description);
                showDialog(
                  context: context,
                  builder: (context) => AlertDialog(
                    title: const Text('AI Guidance'),
                    content: SingleChildScrollView(child: Text(guidance)),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text('Close'),
                      ),
                    ],
                  ),
                );
              },
            ),
            OutlinedButton.icon(
              icon: const Icon(Icons.edit_note, size: 16),
              label: const Text('Reflection'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.blue,
                side: BorderSide(color: Colors.blue.shade200),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (context) => _AddReflectionDialog(goalId: goal.id),
                );
              },
            ),
            OutlinedButton.icon(
              icon: const Icon(Icons.delete, size: 16),
              label: const Text('Remove'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.red,
                side: BorderSide(color: Colors.red.shade200),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
              onPressed: () => provider.removeGoal(goal.id),
            ),
          ],
        ),
      ],
    );
  }
}

// Reflection Dialog
class _AddReflectionDialog extends StatefulWidget {
  final String goalId;

  const _AddReflectionDialog({required this.goalId});

  @override
  State<_AddReflectionDialog> createState() => _AddReflectionDialogState();
}

class _AddReflectionDialogState extends State<_AddReflectionDialog> {
  final _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Add Reflection',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _controller,
              maxLines: 4,
              decoration: InputDecoration(
                hintText: 'What did you learn? How did you feel?',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepPurple,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onPressed: () {
                    if (_controller.text.isNotEmpty) {
                      Provider.of<GoalProvider>(context, listen: false)
                          .addReflection(widget.goalId, _controller.text);
                      Navigator.pop(context);
                    }
                  },
                  child: const Text(
                    'Save',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
