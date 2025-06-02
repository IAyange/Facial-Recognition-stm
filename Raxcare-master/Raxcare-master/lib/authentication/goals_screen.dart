import 'package:flutter/material.dart';
import 'support_system_screen.dart';

class GoalsScreen extends StatefulWidget {
  const GoalsScreen({super.key});

  @override
  State<GoalsScreen> createState() => _GoalsScreenState();
}

class _GoalsScreenState extends State<GoalsScreen> {
  final Set<String> _selectedGoals = {};

  final List<Map<String, dynamic>> _commonGoals = [
    {'text': 'Stay sober', 'icon': Icons.local_drink_outlined},
    {'text': 'Avoid triggers', 'icon': Icons.warning_amber_outlined},
    {'text': 'Build healthy habits', 'icon': Icons.health_and_safety_outlined},
    {'text': 'Improve relationships', 'icon': Icons.people_outline},
    {'text': 'Manage stress', 'icon': Icons.self_improvement_outlined},
    {'text': 'Find purpose', 'icon': Icons.emoji_objects_outlined},
    {'text': 'Physical health', 'icon': Icons.fitness_center_outlined},
    {'text': 'Mental wellness', 'icon': Icons.psychology_outlined},
    {'text': 'Better sleep', 'icon': Icons.bedtime_outlined},
    {'text': 'Financial stability', 'icon': Icons.attach_money_outlined},
    {'text': 'Spiritual growth', 'icon': Icons.spa_outlined},
    {'text': 'Career goals', 'icon': Icons.work_outline_outlined},
  ];

  void _toggleGoal(String goal) {
    setState(() {
      if (_selectedGoals.contains(goal)) {
        _selectedGoals.remove(goal);
      } else {
        _selectedGoals.add(goal);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final gridHeight = screenHeight * 0.6; // Adjust this ratio as needed

    return Scaffold(
      appBar: AppBar(
        title: const Text("My Recovery Goals"),
        centerTitle: true,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),
            Text(
              "Select your recovery goals",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              "Choose all that apply to you",
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey.shade600,
                  ),
            ),
            const SizedBox(height: 16),

            // Grid of goal options
            SizedBox(
              height: gridHeight,
              child: GridView.count(
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 3,
                childAspectRatio: 1.1,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                children: _commonGoals.map((goal) {
                  final isSelected = _selectedGoals.contains(goal['text']);
                  return GestureDetector(
                    onTap: () => _toggleGoal(goal['text']),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? Theme.of(context)
                                .colorScheme
                                .primary
                                .withOpacity(0.1)
                            : Colors.grey.shade100,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: isSelected
                              ? Theme.of(context).colorScheme.primary
                              : Colors.transparent,
                          width: 2,
                        ),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            goal['icon'],
                            size: 28,
                            color: isSelected
                                ? Theme.of(context).colorScheme.primary
                                : Colors.grey.shade700,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            goal['text'],
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                              color: isSelected
                                  ? Theme.of(context).colorScheme.primary
                                  : Colors.grey.shade800,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const SupportSystemScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text("Continue"),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
