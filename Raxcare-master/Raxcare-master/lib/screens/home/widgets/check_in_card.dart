import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../form/relapseform.dart';
import '../controllers/home_controller.dart';

// State interface for CheckInCard
abstract class CheckInState {
  Widget buildContent(BuildContext context, _CheckInCardState state);
  void onCheckIn(BuildContext context, _CheckInCardState state);
  void onRelapse(BuildContext context, _CheckInCardState state);
  Color getGradientStart();
  Color getGradientEnd();
}

// Default state
class ActiveCheckInState implements CheckInState {
  @override
  Widget buildContent(BuildContext context, _CheckInCardState state) {
    final daysSober = state.widget.controller.getDaysSober();

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('Daily Check-In',
            style: GoogleFonts.poppins(
                fontSize: 22,
                fontWeight: FontWeight.w700,
                color: Colors.white)),
        const SizedBox(height: 16),

        // Larger check-in button with countdown
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              ElevatedButton(
                onPressed: state.widget.controller.hasCheckedInToday
                    ? null
                    : () => state.handleCheckIn(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: getGradientEnd(),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  shape: const StadiumBorder(),
                  elevation: 0,
                ),
                child: Text(
                  state.widget.controller.hasCheckedInToday
                      ? '✓ CHECKED IN'
                      : 'CHECK IN NOW',
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const SizedBox(height: 8),

              // Countdown timer
              if (state.widget.controller.hasCheckedInToday)
                StreamBuilder<DateTime>(
                  stream: Stream.periodic(
                      const Duration(seconds: 1), (_) => DateTime.now()),
                  builder: (context, snapshot) {
                    final now = DateTime.now();
                    final tomorrow = DateTime(now.year, now.month, now.day + 1);
                    final remaining = tomorrow.difference(now);

                    return Column(
                      children: [
                        Text(
                          'Next check-in available in:',
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: Colors.white70,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${remaining.inHours.toString().padLeft(2, '0')}h '
                          '${(remaining.inMinutes % 60).toString().padLeft(2, '0')}m '
                          '${(remaining.inSeconds % 60).toString().padLeft(2, '0')}s',
                          style: GoogleFonts.poppins(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    );
                  },
                ),
            ],
          ),
        ),

        const SizedBox(height: 20),
        _buildMilestoneRow(daysSober),

        if (!state.widget.controller.hasCheckedInToday && daysSober > 0)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: Text(
              'Check in daily to track your progress!',
              style: GoogleFonts.poppins(
                  fontSize: 14,
                  color: Colors.white70,
                  fontStyle: FontStyle.italic),
            ),
          ),
        const SizedBox(height: 12),

        // Relapse button
        OutlinedButton(
          onPressed: () => state.handleRelapse(),
          style: OutlinedButton.styleFrom(
            foregroundColor: Colors.white70,
            side: const BorderSide(color: Colors.white70),
            shape: const StadiumBorder(),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
          ),
          child: Text(
            'Report Relapse',
            style: GoogleFonts.poppins(
              fontSize: 14,
              decoration: TextDecoration.underline,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMilestoneRow(int days) => Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _statItem('DAYS', days.toString()),
            _statItem('WEEKS', (days / 7).floor().toString()),
            _statItem('MONTHS', (days / 30).floor().toString()),
          ],
        ),
      );

  Widget _statItem(String label, String value) => Column(
        children: [
          Text(value,
              style: GoogleFonts.poppins(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.white)),
          const SizedBox(height: 4),
          Text(label,
              style: GoogleFonts.poppins(
                fontSize: 12,
                color: Colors.white70,
                letterSpacing: 1.2,
              )),
        ],
      );

  @override
  void onCheckIn(BuildContext context, _CheckInCardState state) =>
      state.handleCheckIn();

  @override
  void onRelapse(BuildContext context, _CheckInCardState state) =>
      state.handleRelapse();

  @override
  Color getGradientStart() => const Color(0xFF6B48FF);

  @override
  Color getGradientEnd() => const Color(0xFF00DDEB);
}

// Celebration state
class MilestoneCheckInState implements CheckInState {
  @override
  Widget buildContent(BuildContext context, _CheckInCardState state) {
    return Stack(
      alignment: Alignment.center,
      children: [
        ActiveCheckInState().buildContent(context, state),
        const Positioned(
          top: 8,
          right: 8,
          child: Icon(Icons.star, color: Colors.yellowAccent, size: 24),
        ),
      ],
    );
  }

  @override
  void onCheckIn(BuildContext context, _CheckInCardState state) =>
      ActiveCheckInState().onCheckIn(context, state);

  @override
  void onRelapse(BuildContext context, _CheckInCardState state) =>
      ActiveCheckInState().onRelapse(context, state);

  @override
  Color getGradientStart() => const Color(0xFFFF7E5F);

  @override
  Color getGradientEnd() => const Color(0xFFfeb47b);
}

class CheckInCard extends StatefulWidget {
  final HomeController controller;
  final VoidCallback onCheckIn;

  const CheckInCard({
    Key? key,
    required this.controller,
    required this.onCheckIn,
  }) : super(key: key);

  @override
  State<CheckInCard> createState() => _CheckInCardState();
}

class _CheckInCardState extends State<CheckInCard> {
  CheckInState state = ActiveCheckInState();

  @override
  void initState() {
    super.initState();
    _updateState();
    widget.controller.addListener(_updateUI);
  }

  void _updateState() {
    state = widget.controller.shouldCelebrate()
        ? MilestoneCheckInState()
        : ActiveCheckInState();
  }

  void _updateUI() {
    setState(_updateState);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_updateUI);
    super.dispose();
  }

  Future<void> handleCheckIn() async {
    if (await _showPrompt(
        'Check-In Time', 'Are you ready to mark today as a win?', 'Yes')) {
      await widget.controller.checkIn(context);
      widget.onCheckIn();
      if (widget.controller.shouldCelebrate()) {
        setState(() => state = MilestoneCheckInState());
        await _showPrompt('Milestone Achieved! 🎉',
            'You\'ve hit ${widget.controller.getDaysSober()} days!', 'Awesome');
      }
    }
  }

  Future<void> handleRelapse() async {
    if (await _showPrompt(
        'Relapse?', 'Want to log this to grow stronger?', 'Yes')) {
      await _showPrompt(
          'You\'ve Got This!', 'Every step teaches us something.', 'Continue');
      await widget.controller.handleRelapse(context);
      Navigator.push(
          context, MaterialPageRoute(builder: (_) => RelapseFormScreen()));
      setState(() => state = ActiveCheckInState());
    }
  }

  Future<bool> _showPrompt(
      String title, String message, String confirmText) async {
    return (await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            backgroundColor: Colors.white,
            title: Text(title,
                style: GoogleFonts.poppins(
                    fontWeight: FontWeight.bold,
                    color: state.getGradientEnd())),
            content: Text(message,
                style: GoogleFonts.poppins(color: Colors.black87)),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: Text('Cancel',
                    style: GoogleFonts.poppins(color: Colors.grey)),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(context, true),
                style: ElevatedButton.styleFrom(
                  backgroundColor: state.getGradientEnd(),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                ),
                child: Text(confirmText,
                    style: GoogleFonts.poppins(color: Colors.white)),
              ),
            ],
          ),
        )) ??
        false;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [state.getGradientStart(), state.getGradientEnd()],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: const [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 12,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: state.buildContent(context, this),
    );
  }
}
