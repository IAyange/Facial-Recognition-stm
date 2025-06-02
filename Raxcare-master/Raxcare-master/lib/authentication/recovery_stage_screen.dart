import 'package:flutter/material.dart';
import '../utils/preferences_service.dart';
import 'goals_screen.dart';

class RecoveryStageScreen extends StatefulWidget {
  const RecoveryStageScreen({super.key});

  @override
  State<RecoveryStageScreen> createState() => _RecoveryStageScreenState();
}

class _RecoveryStageScreenState extends State<RecoveryStageScreen> {
  String _selectedStage = 'Just Starting';
  bool _isSubmitting = false;

  final List<Map<String, dynamic>> _recoveryStages = [
    {
      'title': 'Just Starting',
      'description': 'Beginning my recovery journey',
      'icon': Icons.flag_outlined,
    },
    {
      'title': 'In Treatment',
      'description': 'Currently in a treatment program',
      'icon': Icons.medical_services_outlined,
    },
    {
      'title': 'Post-Treatment',
      'description': 'Completed treatment, maintaining recovery',
      'icon': Icons.celebration_outlined,
    },
  ];

  Future<void> _submitRecoveryStage() async {
    setState(() => _isSubmitting = true);
    await PreferencesService.saveData('recovery_stage', _selectedStage);
    if (!mounted) return;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const GoalsScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Recovery Progress"),
        centerTitle: true,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 32),
            Text(
              "Where are you in your recovery?",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              "This helps us tailor support to your needs",
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey.shade600,
                  ),
            ),
            const SizedBox(height: 32),

            // Stage selection cards
            Column(
              children: _recoveryStages.map((stage) {
                final isSelected = _selectedStage == stage['title'];
                return Card(
                  margin: const EdgeInsets.only(bottom: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(
                      color: isSelected
                          ? Theme.of(context).colorScheme.primary
                          : Colors.transparent,
                      width: 2,
                    ),
                  ),
                  elevation: isSelected ? 4 : 0,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () {
                      setState(() {
                        _selectedStage = stage['title'];
                      });
                    },
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        children: [
                          Icon(
                            stage['icon'],
                            color: isSelected
                                ? Theme.of(context).colorScheme.primary
                                : Colors.grey,
                            size: 28,
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  stage['title'],
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: isSelected
                                        ? Theme.of(context).colorScheme.primary
                                        : Colors.black,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  stage['description'],
                                  style: TextStyle(
                                    color: Colors.grey.shade600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          if (isSelected)
                            Icon(
                              Icons.check_circle,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                        ],
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
            const Spacer(),

            // Next button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isSubmitting ? null : _submitRecoveryStage,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isSubmitting
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text("Continue"),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
